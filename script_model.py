from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

SIMILALITY_HIGH = 0.85
SIMILALITY_MEDIUM = 0.65
# 임계값 조절하기~~

def analyse_script_model(script: str, full_text: str) -> dict:
    if not script or not full_text:
        return {
            "similarity_score" : 0,
            "level": 0,
            "feedbacks": ["대본 또는 발화 텍스트가 없어 분석할 수 없음"]
        }
    
    embeddings = model.encode([script, full_text])
    score = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
    score = max(0.0, min(1.0, score))

    level, feedbacks = get_feedback(score)
    return{
        "similarity_score": round(score*100),
        "level": level,
        "feedbacks": feedbacks
    }

def get_feedback(score: float) -> tuple[str, list[str]]:
    if score >= SIMILALITY_HIGH:
        return 1, ["대본 내용과 유사"]
    elif score >= SIMILALITY_MEDIUM:
        return 2, ["대본 내용에서 일부 누락된 키워드가 있음"]
    else:
        return 3, ["핵심 내용을 발화하지 못 함"]
