def get_mean_scores(scores: dict, name: str) -> dict:
    for i in scores.keys():
        scores[i] = sum(scores[i])/len(scores[i])
    scores["name"] = name
    return scores