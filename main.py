import asyncio
import base64
import io
import random
import re
from datetime import datetime
from pathlib import Path

import aiohttp
from PIL import Image as PILImage

import astrbot.core.message.components as Comp
from astrbot.api import logger
from astrbot.api.event import filter
from astrbot.api.star import Context, Star, StarTools, register
from astrbot.core import AstrBotConfig
from astrbot.core.message.components import Image
from astrbot.core.platform.astr_message_event import AstrMessageEvent


class ImageWorkflow:
    def __init__(self):
        self.session = None

    async def get_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session

    async def _download_image(self, url: str) -> bytes | None:
        try:
            session = await self.get_session()
            async with session.get(url) as resp:
                resp.raise_for_status()
                return await resp.read()
        except Exception as e:
            logger.error(f"图片下载失败: {e}")
            return None

    async def _get_avatar(self, user_id: str) -> bytes | None:
        if not user_id.isdigit():
            user_id = "".join(random.choices("0123456789", k=9))
        avatar_url = f"https://q4.qlogo.cn/headimg_dl?dst_uin={user_id}&spec=640"
        try:
            session = await self.get_session()
            async with session.get(avatar_url, timeout=10) as resp:
                resp.raise_for_status()
                return await resp.read()
        except Exception as e:
            logger.error(f"下载头像失败: {e}")
            return None

    def _extract_first_frame_sync(self, raw: bytes) -> bytes:
        """
        使用PIL库处理图片数据。如果是GIF，则提取第一帧并转为PNG。
        """
        img_io = io.BytesIO(raw)
        img = PILImage.open(img_io)
        if img.format != "GIF":
            return raw
        logger.info("检测到GIF, 将抽取 GIF 的第一帧来生图")
        first_frame = img.convert("RGBA")
        out_io = io.BytesIO()
        first_frame.save(out_io, format="PNG")
        return out_io.getvalue()

    async def _load_bytes(self, src: str) -> bytes | None:
        raw: bytes | None = None
        loop = asyncio.get_running_loop()

        if Path(src).is_file():
            raw = await loop.run_in_executor(None, Path(src).read_bytes)
        elif src.startswith("http"):
            raw = await self._download_image(src)
        elif src.startswith("base64://"):
            raw = await loop.run_in_executor(None, base64.b64decode, src[9:])

        if not raw:
            return None
        return await loop.run_in_executor(None, self._extract_first_frame_sync, raw)

    async def get_first_image(self, event: AstrMessageEvent) -> bytes | None:
        for s in event.message_obj.message:
            if isinstance(s, Comp.Reply) and s.chain:
                for seg in s.chain:
                    if isinstance(seg, Comp.Image):
                        if seg.url and (img := await self._load_bytes(seg.url)):
                            return img
                        if seg.file and (img := await self._load_bytes(seg.file)):
                            return img
        for seg in event.message_obj.message:
            if isinstance(seg, Comp.Image):
                if seg.url and (img := await self._load_bytes(seg.url)):
                    return img
                if seg.file and (img := await self._load_bytes(seg.file)):
                    return img
            elif isinstance(seg, Comp.At):
                if avatar := await self._get_avatar(str(seg.qq)):
                    return avatar
        return await self._get_avatar(event.get_sender_id())

    async def terminate(self):
        if self.session and not self.session.closed:
            await self.session.close()


@register(
    "astrbot_plugin_figurine_workshop",
    "gufei233",  # 改成您的名字
    "使用 Gemini API 将图片手办化 (支持 nano-banana)",
    "1.1.0",  # 更新版本号
)
class FigurineWorkshopPlugin(Star):  # 改名为更合适的类名
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.conf = config
        self.save_image = config.get("save_image", True)
        self.plugin_data_dir = StarTools.get_data_dir(
            "astrbot_plugin_figurine_workshop"
        )
        self.api_keys = self.conf.get("gemini_api_keys", [])
        self.current_key_index = 0
        self.api_base_url = self.conf.get(
            "api_base_url", "https://generativelanguage.googleapis.com"
        )
        self.figurine_style = self.conf.get("figurine_style", "deluxe_box")
        self.model_name = self.conf.get("model_name", "gemini-2.0-flash-preview-image-generation")
        self.image_workflow = ImageWorkflow()
        
        # 确保数据目录存在
        self.plugin_data_dir.mkdir(exist_ok=True)

    def on_star_shutdown(self):
        logger.info("手办工坊插件正在关闭...")
        asyncio.create_task(self.image_workflow.terminate())

    @filter.command(["手办化", "手辦化", "/figure", "/figurine"])
    async def handle_figurine(self, event: AstrMessageEvent):
        """处理手办化请求"""
        try:
            logger.info("收到手办化请求")
            
            # 检查是否配置了API密钥
            if not self.api_keys:
                await event.reply(
                    "❌ 未配置 Gemini API 密钥。\n请在插件配置中添加至少一个 API 密钥。\n获取地址: https://aistudio.google.com/"
                )
                return

            # 发送处理提示
            style_name = (
                "豪华盒装版" if self.figurine_style == "deluxe_box" else "经典版"
            )
            await event.reply(f"🎨 正在生成{style_name}手办...")

            # 获取图片
            image_bytes = await self.image_workflow.get_first_image(event)
            if not image_bytes:
                await event.reply("❌ 未找到可处理的图片")
                return

            # 生成手办
            result = await self.generate_figurine(event, image_bytes)

            if result:
                # 发送生成的图片
                await event.reply(Image(file=result))
                logger.info("手办生成成功")
            else:
                await event.reply("❌ 生成失败，请稍后重试")

        except Exception as e:
            logger.error(f"处理手办化请求时出错: {e}", exc_info=True)
            await event.reply(f"❌ 发生错误: {str(e)}")

    async def _extract_image_from_response(self, data: dict) -> bytes | None:
        """从响应中提取图片数据（支持多种格式）"""
        if "candidates" not in data or not data["candidates"]:
            logger.error("响应中没有 candidates")
            return None
        
        for candidate in data["candidates"]:
            if "content" not in candidate or "parts" not in candidate["content"]:
                continue
            
            for part in candidate["content"]["parts"]:
                # 方式1：直接的 base64 图片数据（官方 API 和部分第三方）
                if "inlineData" in part and "data" in part["inlineData"]:
                    logger.info("找到 inlineData 格式的图片")
                    return base64.b64decode(part["inlineData"]["data"])
                
                # 方式2：文本中的图片链接（nano-banana 等）
                if "text" in part:
                    text = part["text"]
                    
                    # 提取所有可能的图片 URL
                    urls = []
                    
                    # Markdown 格式: ![...](URL)
                    urls.extend(re.findall(r'!$$.*?$$$(https?://[^$]+)\)', text))
                    
                    # 下载链接格式: [下载...](URL)
                    urls.extend(re.findall(r'$$下载.*?$$$(https?://[^$]+)\)', text))
                    
                    # 直接的 URL（以常见图片扩展名结尾）
                    urls.extend(re.findall(r'https?://[^\s<>"{}|\\^`$$$$]+\.(?:png|jpg|jpeg|gif|webp)', text))
                    
                    if urls:
                        logger.info(f"找到 {len(urls)} 个图片链接")
                        
                        # 尝试下载图片
                        session = await self.image_workflow.get_session()
                        for url in urls:
                            try:
                                logger.info(f"尝试下载: {url}")
                                async with session.get(url, timeout=30) as img_response:
                                    if img_response.status == 200:
                                        image_data = await img_response.read()
                                        logger.info("成功从 URL 下载图片")
                                        return image_data
                            except Exception as e:
                                logger.warning(f"下载图片失败 {url}: {e}")
                                continue
        
        logger.error("未能从响应中提取到图片")
        return None

    async def generate_figurine(self, event, image_bytes):
        """生成手办风格图片"""
        # 将图片编码为base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # 获取提示词
        prompts = self.conf.get("prompts", {})
        if self.figurine_style not in prompts:
            logger.error(f"未找到风格 {self.figurine_style} 的提示词")
            return None

        prompt_text = prompts[self.figurine_style]

        # 构建请求数据
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt_text},
                        {
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": image_base64,
                            }
                        },
                    ],
                }
            ],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
        }

        # 获取会话
        session = await self.image_workflow.get_session()

        # 尝试不同的 API Key
        for i in range(len(self.api_keys)):
            key_index = (self.current_key_index + i) % len(self.api_keys)
            current_key = self.api_keys[key_index]

            try:
                logger.info(
                    f"使用 API Key {key_index + 1}/{len(self.api_keys)}"
                )

                # 构建请求URL
                base_url = self.api_base_url.strip().rstrip("/")
                
                # 根据 URL 判断使用 v1 还是 v1beta
                if "generativelanguage.googleapis.com" in base_url:
                    endpoint = f"{base_url}/v1beta/models/{self.model_name}:generateContent?key={current_key}"
                else:
                    # 第三方平台可能使用 v1
                    endpoint = f"{base_url}/v1/models/{self.model_name}:generateContent?key={current_key}"

                headers = {"Content-Type": "application/json"}

                # 发送请求
                async with session.post(
                    url=endpoint, json=payload, headers=headers, timeout=60
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # 使用新的提取方法
                        image_data = await self._extract_image_from_response(data)
                        
                        if image_data:
                            if self.save_image:
                                # 保存图片
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                output_filename = f"figurine_{event.get_sender_id()}_{timestamp}.png"
                                output_path = self.plugin_data_dir / output_filename
                                
                                with open(output_path, "wb") as f:
                                    f.write(image_data)
                                
                                logger.info(f"手办图片已保存: {output_path}")
                                return str(output_path)
                            else:
                                # 直接返回base64编码的图片
                                temp_path = self.plugin_data_dir / "temp_figurine.png"
                                with open(temp_path, "wb") as f:
                                    f.write(image_data)
                                return str(temp_path)
                        else:
                            logger.error("响应中未找到图片数据")

                    elif response.status == 429:
                        logger.warning(f"API Key {key_index + 1} 达到速率限制")
                        continue
                    else:
                        error_text = await response.text()
                        logger.error(f"API 错误 ({response.status}): {error_text}")
                        
                        # 如果是配额错误，尝试下一个 key
                        if "RESOURCE_EXHAUSTED" in error_text or "quota" in error_text.lower():
                            continue
                        
            except asyncio.TimeoutError:
                logger.error("请求超时")
                continue
            except Exception as e:
                logger.error(f"生成手办失败: {e}", exc_info=True)
                continue

        # 更新当前使用的 key 索引
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        return None

    @filter.command(["手办帮助", "手办化帮助", "/figurine_help"])
    async def show_help(self, event: AstrMessageEvent):
        """显示帮助信息"""
        help_text = f"""🎭 **手办工坊插件帮助**

**使用方法：**
1. 发送图片 + "手办化" 
2. 回复图片消息 + "手办化"
3. @某人 + "手办化" (使用其头像)

**可用命令：**
- 手办化 / 手辦化 / /figure / /figurine
- 手办帮助 / /figurine_help

**支持的模型：**
- gemini-2.0-flash-preview-image-generation (推荐)
- nano-banana (Gemini 2.5 Flash Image)

**当前配置：**
- 风格：{'豪华盒装版' if self.figurine_style == 'deluxe_box' else '经典版'}
- 模型：{self.model_name}
- API Keys：{len(self.api_keys)}个

获取 API Key: https://aistudio.google.com/"""
        
        await event.reply(help_text)
