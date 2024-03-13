import re

unparsed_questions = """Sure, I can help with that!\n \
    * What was the companies revenue in 2022 (AMD_2022_10K)\n \
    * What was the companies net income in 2021 (INTC_2021_10Q_1)\n \
    """
tools = ["AMD_2022_10K", "INTC_2021_10Q_1", "AMD_2021_10K"]
leftovers = []
lines = unparsed_questions.split("\n")
for line in lines:
    for report in tools:
        if report in line:
            r_match = [r_part for r_part in report.split("_")]
            question = re.sub(rf".{re.escape(report)}.", "", line)
            r_match.append(question)
            leftovers.append(tuple(r_match))
print(leftovers)