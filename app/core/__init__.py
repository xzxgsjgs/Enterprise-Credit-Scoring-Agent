"""app.core —— Streamlit 与底层 tools 之间的编排层。"""
from app.core.training import run_training_pipeline, TrainingResult

__all__ = ["run_training_pipeline", "TrainingResult"]
