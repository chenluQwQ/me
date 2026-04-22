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
from astrbot.api.provider import ProviderRequest
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
    "https://github.com/chenluQwQ/astrbot_plugin_imagegen",
)
class ImageGenPlugin(Star):

    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        self.api_base = config.get("api_base", "").rstrip("/")
        self.api_key = config.get("api_key", "")
        self.default_size = config.get("default_size", "1024x1024")
        self.default_quality = config.get("default_quality", "auto")
        self.optimize = config.get("optimize_prompt", True)
        self.enable_tool = config.get("enable_llm_tool", True)
        self.output_format = config.get("output_format", "png")

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

        yield event.plain_result("🎨 正在生成，请稍候…")

        # 优化提示词
        final_prompt = prompt
        if self.optimize:
            optimized = await self._optimize_prompt(prompt)
            if optimized:
                final_prompt = optimized
                logger.info(f"[ImageGen] 优化提示词: {prompt} → {final_prompt[:80]}...")

        # 调 API
        img_bytes = await self._call_api(final_prompt)
        if img_bytes is None:
            yield event.plain_result("❌ 生图失败，请检查日志。")
            return

        yield event.image_result(img_bytes)

    # ── LLM 工具调用 ────────────────────────────────────────

    @filter.llm_tool(name="generate_image")
    async def tool_generate_image(
        self, event: AstrMessageEvent, prompt: str, size: str = "", quality: str = ""
    ):
        """根据描述生成图片。当用户想要你画画、生成图片、创建图像时调用此工具。

        Args:
            prompt(string): 图片描述，尽量用英文，包含主体、风格、光影等信息
            size(string): 可选尺寸：1024x1024 / 1536x1024 / 1024x1536 / auto
            quality(string): 可选质量：low / medium / high / auto
        """
        if not self.enable_tool:
            yield event.plain_result("生图工具未启用。")
            return

        if not self.api_base or not self.api_key:
            yield event.plain_result("生图插件未配置 API。")
            return

        # 工具调用时也优化一下
        final_prompt = prompt
        if self.optimize:
            optimized = await self._optimize_prompt(prompt)
            if optimized:
                final_prompt = optimized

        img_bytes = await self._call_api(
            final_prompt,
            size=size or self.default_size,
            quality=quality or self.default_quality,
        )
        if img_bytes is None:
            yield event.plain_result("生图失败，请稍后再试。")
            return

        yield event.image_result(img_bytes)

    # ── 内部方法 ────────────────────────────────────────────

    async def _optimize_prompt(self, user_text: str) -> str | None:
        """用 LLM 把用户的随意描述优化成英文生图提示词"""
        try:
            provider_req = ProviderRequest(prompt="", session_id="_imagegen_opt")
            provider_req.system_message = PROMPT_OPTIMIZE_SYSTEM
            provider_req.prompt = user_text

            resp = await self.context.llm_generate(provider_req)
            if resp and resp.completion_text:
                return resp.completion_text.strip()
        except Exception as e:
            logger.warning(f"[ImageGen] 提示词优化失败，使用原始文本: {e}")
        return None

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
            timeout = aiohttp.ClientTimeout(total=180)  # 复杂提示词可能要 2 分钟
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=body, headers=headers) as resp:
                    if resp.status != 200:
                        err = await resp.text()
                        logger.error(f"[ImageGen] API 返回 {resp.status}: {err[:300]}")
                        return None
                    data = await resp.json()

            b64 = data.get("data", [{}])[0].get("b64_json")
            if not b64:
                logger.error("[ImageGen] API 返回中无 b64_json")
                return None

            return base64.b64decode(b64)

        except asyncio.TimeoutError:
            logger.error("[ImageGen] API 请求超时（180s）")
            return None
        except Exception as e:
            logger.error(f"[ImageGen] API 调用异常: {e}")
            return None
