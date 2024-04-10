import time

""" compute the score between expected and given answer """
def compute_score(expected, answer, cross_encoder=None):
    score = None
    start_time = time.time_ns()
    if cross_encoder:
        score = cross_encoder.predict([expected, answer])
    stop_time = time.time_ns()
    duration = stop_time - start_time
    duration_s = duration/1e9
    return score, duration_s