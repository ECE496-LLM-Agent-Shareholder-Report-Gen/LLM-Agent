import datetime
import json

""" Session
holds the state of the session, which represents
a user-LLM interaction """
class Session:

    def __init__(self, name=None, llm_chain=None, retrieval_strategy=None, conversation_history=[], reports=[], memory_enabled=False, k=None, k_i=None):
        self.conversation_history = conversation_history
        self.reports = reports
        self.llm_chain = llm_chain
        self.retrieval_strategy = retrieval_strategy
        if name == None:
            now = datetime.datetime.now()
            formatted_time = now.strftime("%Y-%m-%dT%H:%M:%S")
            self.name = formatted_time
        else:
            self.name = name
        self.memory_enabled = False
        self.k = k
        self.k_i = k_i
        self.memory_enabled = memory_enabled
    
    def add_to_conversation(self, question, answer):
        qa = QA(question, answer)
        if not isinstance(self.conversation_history, list): 
            self.conversation_history = []
        self.conversation_history.append(qa)

    def add_report(self, company, year, report_type, quarter=None, report_type_other=None, file_location=None):
        report = Report(company, year, report_type, quarter, report_type_other, file_location)
        self.reports.append(report)

    def add_reports(self, report_dict_list):
        for report_dict in report_dict_list:
            self.add_report(**report_dict)

    def add_conversation_history(self, conversation_history_dict_list):
        for conversation_history_dict in conversation_history_dict_list:
            self.add_report(**conversation_history_dict)

    def encode(self):
        return vars(self)

    def to_dict(self, indent=None):
        json_str = json.dumps(self, default=lambda o: o.encode(), indent=indent)
        return json.loads(json_str)
    
    @classmethod
    def from_dict(cls, cls_dict):
        ses = cls(**cls_dict)
        ses.reports = []
        ses.conversation_history = []
        ses.add_reports(cls_dict['reports'])
        ses.add_conversation_history(cls_dict['conversation_history'])
        return ses


""" QA
class that represents a question and an answer """
class QA:
    def __init__(self, question, answer):
        self.question = question
        self.answer = answer

    def encode(self):
        return vars(self)


""" Report
class that represents a company's shareholder report, metadata ONLY """
class Report:
    def __init__(self, company, year, report_type, quarter=None, report_type_other=None, file_location=None):
        self.company = company
        self.year = year
        self.report_type = report_type
        if report_type.lower() == '10q':
            assert quarter != None, "Quarter must exist for report type 10Q"
        if report_type.lower() == 'other':
            assert report_type_other != None, "Report type must be specified if 'Other' is selected"
        self.quarter = quarter
        self.report_type_other = report_type_other
        self.file_location = file_location
    
    def encode(self):
        return vars(self)
