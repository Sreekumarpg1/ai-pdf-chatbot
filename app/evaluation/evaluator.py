from sentence_transformers import util


def evaluate_answer(answer, context, model):
    context_text = " ".join(context)

    similarity = util.cos_sim(
        model.encode(answer),
        model.encode(context_text)
    ).item()

    return round(min(max(similarity, 0), 1), 2)