"""
AstrBot Plugin - ImageGen (GPT-Image-2)
$生图 命令 + LLM 工具调用 + AI 提示词优化
"""
import io
import base64
import asyncio
import aiohttp
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api.message_components import Image, Plain
from astrbot.api import logger

PROMPT_OPTIMIZE_SYSTEM = """你是一个图像生成提示词专家。用户会给你一段描述，你需要将它改写成适合 AI 绘图模型的英文提示词。

规则：
- 输出纯英文，不要中文
- 简洁有力，不超过 150 词
- 包含主体、风格、光影、构图等关键信息
- 不要输出解释，只输出优化后的提示词本身"""


@register(
    "astrbot_plugin_imagegen",
    "chenluQwQ",
    "GPT-Image-2 生图插件 — $生图 命令 + LLM 工具调用 + AI 提示词优化",
    "1.0.0",
    "https://github.com/chenluQwQ/me",
)
class ImageGenPlugin(Star):

    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        self.api_base = config.get("api_base", "").rstrip("/")
        self.api_key = config.get("api_key", "")
        self.default_size = config.get("default_size", "1024x1024")
        self.default_quality = config.get("default_quality", "auto")
        self.enable_tool = config.get("enable_llm_tool", True)
        self.output_format = config.get("output_format", "png")
        # 提示词优化相关
        self.enable_prompt_optimize = config.get("enable_prompt_optimize", True)
        self.optimize_api_base = config.get("optimize_api_base", "").rstrip("/") or self.api_base
        self.optimize_api_key = config.get("optimize_api_key", "") or self.api_key
        self.optimize_model = config.get("optimize_model", "gpt-4o-mini")

    # ── 命令：$生图 ──────────────────────────────────────────

    @filter.command("生图", alias=["画图", "draw"])
    async def cmd_generate(self, event: AstrMessageEvent, *, prompt: str = ""):
        """生成图片。用法：$生图 一只猫在月球上弹吉他"""
        prompt = prompt.strip()
        if not prompt:
            yield event.plain_result("用法：$生图 <描述>\n例：$生图 赛博朋克风格的东京街头夜景")
            return

        if not self.api_base or not self.api_key:
            yield event.plain_result("❌ 插件未配置 API 地址或 Key，请在后台设置。")
            return

        # 提示词优化
        if self.enable_prompt_optimize:
            yield event.plain_result("🎨 正在优化提示词并生成，请稍候…")
            optimized = await self._optimize_prompt(prompt)
            if optimized:
                logger.info(f"[ImageGen] 原始: {prompt}")
                logger.info(f"[ImageGen] 优化: {optimized}")
                prompt = optimized
        else:
            yield event.plain_result("🎨 正在生成，请稍候…")

        # 调 API
        img_bytes = await self._call_api(prompt)
        if img_bytes is None:
            yield event.plain_result("❌ 生图失败，请检查日志。")
            return

        if isinstance(img_bytes, str) and img_bytes.startswith("http"):
            yield event.chain_result([Image.fromURL(img_bytes)])
        else:
            yield event.image_result(img_bytes)

    # ── LLM 工具调用 ────────────────────────────────────────

    @filter.llm_tool(name="generate_image")
    async def tool_generate_image(
        self, event: AstrMessageEvent, prompt: str, size: str = "", quality: str = ""
    ):
        """根据描述生成图片。在以下场景你应该主动调用此工具：
        - 用户明确要求画画、生成图片、创建图像
        - 用户描述了一个视觉场景，配一张图能让回复更生动
        - 用户在聊天中提到想看某个东西的样子
        - 你觉得当前对话配一张图会更有趣或更直观

        Args:
            prompt(string): 图片描述，可以是任何语言，插件会自动优化为专业英文提示词
            size(string): 可选尺寸：1024x1024 / 1536x1024 / 1024x1536 / auto
            quality(string): 可选质量：low / medium / high / auto
        """
        if not self.enable_tool:
            yield event.plain_result("生图工具未启用。")
            return

        if not self.api_base or not self.api_key:
            yield event.plain_result("生图插件未配置 API。")
            return

        # 工具调用也走提示词优化
        if self.enable_prompt_optimize:
            optimized = await self._optimize_prompt(prompt)
            if optimized:
                logger.info(f"[ImageGen] 工具调用原始: {prompt}")
                logger.info(f"[ImageGen] 工具调用优化: {optimized}")
                prompt = optimized

        img_bytes = await self._call_api(
            prompt,
            size=size or self.default_size,
            quality=quality or self.default_quality,
        )
        if img_bytes is None:
            yield event.plain_result("生图失败，请稍后再试。")
            return

        if isinstance(img_bytes, str) and img_bytes.startswith("http"):
            yield event.chain_result([Image.fromURL(img_bytes)])
        else:
            yield event.image_result(img_bytes)

    # ── 提示词优化 ──────────────────────────────────────────

    async def _optimize_prompt(self, raw_prompt: str) -> str | None:
        """调用 LLM 将用户描述优化为专业英文绘图提示词"""
        url = f"{self.optimize_api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.optimize_api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.optimize_model,
            "messages": [
                {"role": "system", "content": PROMPT_OPTIMIZE_SYSTEM},
                {"role": "user", "content": raw_prompt},
            ],
            "max_tokens": 300,
            "temperature": 0.7,
        }

        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=body, headers=headers) as resp:
                    if resp.status != 200:
                        err = await resp.text()
                        logger.warning(f"[ImageGen] 提示词优化失败 {resp.status}: {err[:200]}")
                        return None
                    data = await resp.json()

            choices = data.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "").strip()
                if content:
                    return content
            logger.warning(f"[ImageGen] 优化返回为空: {str(data)[:300]}")
            return None

        except Exception as e:
            logger.warning(f"[ImageGen] 提示词优化异常（将使用原始提示词）: {e}")
            return None

    # ── 图片生成 API ────────────────────────────────────────

    async def _call_api(
        self,
        prompt: str,
        size: str = "",
        quality: str = "",
    ) -> bytes | None:
        """调用 GPT-Image-2 API，返回图片 bytes"""
        url = f"{self.api_base}/images/generations"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": "gpt-image-2",
            "prompt": prompt,
            "n": 1,
            "size": size or self.default_size,
            "quality": quality or self.default_quality,
            "output_format": self.output_format,
        }

        try:
            timeout = aiohttp.ClientTimeout(total=600, sock_read=600, sock_connect=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=body, headers=headers) as resp:
                    logger.info(f"[ImageGen] 收到响应 status={resp.status}")
                    if resp.status != 200:
                        err = await resp.text()
                        logger.error(f"[ImageGen] API 返回 {resp.status}: {err[:300]}")
                        return None
                    data = await resp.json()

            item = (data.get("data") or [{}])[0]
            b64 = item.get("b64_json")
            if b64:
                import tempfile, os
                raw = base64.b64decode(b64)
                fd, fp = tempfile.mkstemp(suffix=".png")
                with os.fdopen(fd, "wb") as f:
                    f.write(raw)
                return fp
            img_url = item.get("url")
            if img_url:
                return img_url
            logger.error(f"[ImageGen] API 返回中无 b64_json/url, 原始: {str(data)[:400]}")
            return None

        except asyncio.TimeoutError:
            logger.error("[ImageGen] API 请求超时（600s）")
            return None
        except Exception as e:
            logger.error(f"[ImageGen] API 调用异常: {e}")
            return None
