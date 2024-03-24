

from sentence_transformers import CrossEncoder


expected = "Net income for 2022 was $1.3 billion compared to $3.2 billion in the prior year. The decrease in net income was primarily driven by lower operating income. (page 43)"

answer = """Sure! Based on the excerpts provided, AMD's net income for the year 2022 is $1,320 million, as reported on page 52 of their Form 10-K report.

Here's the source information:

Company: Advanced Micro Devices, Inc. (AMD)
Year: 2022
Report type: Form 10-K
Page number: 52"""

ce_file_path = "/groups/acmogrp/Large-Language-Model-Agent/language_models/cross_encoder/BAAI_bge-reranker-large"
ce = CrossEncoder(model_name=ce_file_path)

print(ce.predict([expected, answer]))