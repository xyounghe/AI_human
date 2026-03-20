import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from logger import logger
from basereal import BaseReal

MODEL_PATH = "./models/Qwen2.5-1.5B-Instruct"

_tokenizer = None
_model = None


def load_local_model():
    """加载离线 Qwen 模型"""
    global _tokenizer, _model
    if _model is not None:
        return _tokenizer, _model

    start = time.perf_counter()
    logger.info("🔄 正在加载本地 Qwen 模型...")

    _tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True
    )
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    end = time.perf_counter()
    logger.info(f"✅ 模型加载完成，用时 {end - start:.2f}s")
    return _tokenizer, _model


def llm_response(message: str, nerfreal: BaseReal):
    tokenizer, model = load_local_model()

    start = time.perf_counter()
    logger.info(f"🧠 输入内容: {message}")

    # ✅ ChatML 格式
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{message}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    # ✅ 关键：只取新生成 token（不会带 system/user）
    output_ids = outputs[0]
    new_tokens = output_ids[inputs.input_ids.shape[1]:]

    result = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # ✅ 额外清洗（防止模型偶尔输出 assistant 前缀）
    if result.lower().startswith("assistant"):
        result = result.split("assistant", 1)[-1].strip(":： \n")

    end = time.perf_counter()
    logger.info(f"🕒 响应耗时: {end - start:.2f}s")
    logger.info(f"💬 模型输出: {result}")

    # ✅ 分句输出给数字人（保持你原逻辑）
    buffer = ""
    for char in result:
        buffer += char
        if char in "，。！？；:,.!?":
            nerfreal.put_msg_txt(buffer.strip())
            buffer = ""

    if buffer:
        nerfreal.put_msg_txt(buffer.strip())