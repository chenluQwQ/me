# astrbot_plugin_imagegen

GPT-Image-2 生图插件，支持命令调用 + LLM 工具调用 + AI 提示词自动优化。

## 功能

- **`$生图 <描述>`** — 直接生成图片（别名：`$画图`、`$draw`）
- **LLM 工具调用** — AI 对话中自动判断是否需要生图，自行调用
- **提示词优化** — 用 LLM 将中文口语描述优化为英文专业提示词

## 配置

| 字段 | 说明 | 默认值 |
|------|------|--------|
| `api_base` | API 地址（不含路径） | — |
| `api_key` | Bearer Token | — |
| `default_size` | 默认尺寸 | `1024x1024` |
| `default_quality` | 默认质量 | `auto` |
| `optimize_prompt` | 是否 LLM 优化提示词 | `true` |
| `enable_llm_tool` | 注册为 LLM 工具 | `true` |
| `output_format` | 输出格式 png/jpeg/webp | `png` |

## 依赖

- aiohttp（AstrBot 通常已内置）
