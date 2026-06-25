from inference_engine import LocalGGUFEngine
import os

model_path = os.path.join("models", "qwen1.5b-q8.gguf")
engine = LocalGGUFEngine(model_path)

print("Starting test inference...")
result = engine.generate(
    system_prompt="You are a helpful assistant.",
    user_prompt="Say hello in one word.",
    max_tokens=10,
    temperature=0.1
)
print(f"\nFinal Result: {result}")
