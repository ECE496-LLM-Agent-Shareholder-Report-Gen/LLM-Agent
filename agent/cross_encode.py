import time

""" compute the score between expected and given answer """
def compute_score(expected, answer, cross_encoder=None):
    """
    Computes a score based on the expected and actual answers.

    Args:
        expected (str): The expected answer.
        answer (str): The actual answer.
        cross_encoder (CrossEncoder, optional): A cross-encoder model for scoring (if available).

    Returns:
        tuple: A tuple containing the score (float) and duration in seconds (float).
    """
    score = None
    start_time = time.time_ns()
    if cross_encoder:
        score = cross_encoder.predict([expected, answer])
    stop_time = time.time_ns()
    duration = stop_time - start_time
    duration_s = duration/1e9
    return score, duration_s