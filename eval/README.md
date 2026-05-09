# Evaluation

Each subfolder holds one evaluator: `eval_<model>.py` plus a matching
`eval_<model>.sh` launcher. All paths and API keys are passed through
environment variables; every launcher prints the variables it requires when
any are missing.

## Supported Models

| Folder                       | Model                            | Backend                      |
|------------------------------|----------------------------------|------------------------------|
| `gemini/`                    | Gemini 2.x / 3.x family          | Remote API                   |
| `minicpm_o/`                 | MiniCPM-o 4.5                    | OpenAI-compatible server     |
| `qwen25_omni/`               | Qwen2.5-Omni                     | OpenAI-compatible server     |
| `qwen3_omni/`                | Qwen3-Omni                       | OpenAI-compatible server     |
| `baichuan_omni/`             | Baichuan-Omni-1.5                | Local HuggingFace            |
| `gemma4/`                    | Gemma 4                          | Local HuggingFace            |
| `humanomni/`                 | HumanOmni                        | Local HuggingFace            |
| `ming_flash_omni/`           | Ming-Flash-Omni-2.0              | Local HuggingFace            |
| `ming_flash_omni_noaudio/`   | Ming-Flash-Omni-2.0 (visual-only ablation) | Local HuggingFace  |
| `omnivinci/`                 | OmniVinci                        | Local HuggingFace            |
| `qwen2_audio/`               | Qwen2-Audio                      | Local HuggingFace            |
| `qwen3_vl/`                  | Qwen3-VL-32B                     | Local HuggingFace            |
| `qwen3_vl_8b/`               | Qwen3-VL-8B                      | Local HuggingFace            |
| `videollama2/`               | VideoLLaMA 2.1-AV                | Local HuggingFace            |
| `video_salmonn2/`            | Video-SALMONN 2                  | Local HuggingFace            |

## Usage

### Remote API

```bash
export BENCHMARK_DIR=$(pwd)/data
export GEMINI_API_KEY=<your_key>
bash eval/gemini/eval_gemini.sh
```

### OpenAI-compatible server

Host the model yourself (e.g. with vLLM), then:

```bash
export BENCHMARK_DIR=$(pwd)/data
export LVBENCH_BASE_URL=http://127.0.0.1:8000
bash eval/qwen3_omni/eval_qwen3_omni.sh
```

### Local HuggingFace

```bash
export <PREFIX>_MODEL_PATH=/path/to/weights
export <PREFIX>_CLEANED_DIR=$(pwd)/data
export <PREFIX>_VIDEOS_DIR=/path/to/videos
bash eval/<model>/eval_<model>.sh
```
