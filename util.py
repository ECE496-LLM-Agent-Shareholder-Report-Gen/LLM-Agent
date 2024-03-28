import os
import json
import glob
from pathlib import Path
import shutil

from session import Session


""" Company Directory
manages the company directory structure of:
        {path}/company/year. Will hold the following dict:
mydict = {
    "AMD": {
        "2022": {
            "10Q": {
                "Q1": {
                    "file_loc": "./content/companies/AMD/2022/10Q/Q1/AMD_2022_10Q_Q1.pdf",
                    "index_loc": "./content/companies/AMD/2022/10Q/Q1/index"
                },
                "Q2": {
                    "file_loc": "./content/companies/AMD/2022/10Q/Q2/AMD_2022_10Q_Q2.pdf",
                    "index_loc": "./content/companies/AMD/2022/10Q/Q2/index"
                },
                "Q3": {
                    "file_loc": "./content/companies/AMD/2022/10Q/Q3/AMD_2022_10Q_Q3.pdf",
                    "index_loc": "./content/companies/AMD/2022/10Q/Q3/index"
                },
            },
            "10K":  {
                "file_loc": "./content/companies/AMD/2022/10K/AMD_2022_10K.pdf",
                "index_loc": "./content/companies/AMD/2022/10K/index"
            },
            "other": {
                "report_type": "<report_type>",
                "file_loc": "<file_loc>",
                "index_loc": "<index_loc",
            }, 
        },
        "2021": {
            "10Q": {
                "Q1": {
                    "file_loc": "./content/companies/AMD/2021/10Q/Q1/AMD_2021_10Q_Q1.pdf",
                    "index_loc": "./content/companies/AMD/2021/10Q/Q1/index"
                },
                "Q2": {
                    "file_loc": "./content/companies/AMD/2021/10Q/Q2/AMD_2021_10Q_Q2.pdf",
                    "index_loc": "./content/companies/AMD/2021/10Q/Q2/index"
                },
                "Q3": {
                    "file_loc": "./content/companies/AMD/2021/10Q/Q3/AMD_2021_10Q_Q3.pdf",
                    "index_loc": "./content/companies/AMD/2021/10Q/Q3/index"
                },
            },
            "10K":  {
                "file_loc": "./content/companies/AMD/2021/10K/AMD_2021_10K.pdf",
                "index_loc": "./content/companies/AMD/2021/10K/index"
            },
            "other": {
                "report_type": "<report_type>",
                "file_loc": "<file_loc>",
                "index_loc": "<index_loc",
            }, 
        }
    },
    "INTC": {
        "2022": {
            "10Q": {
                "Q1": {
                    "file_loc": "./content/companies/INTC/2022/10Q/Q1/INTC_2022_10Q_Q1.pdf",
                    "index_loc": "./content/companies/INTC/2022/10Q/Q1/index"
                },
                "Q2": {
                    "file_loc": "./content/companies/INTC/2022/10Q/Q2/INTC_2022_10Q_Q2.pdf",
                    "index_loc": "./content/companies/INTC/2022/10Q/Q2/index"
                },
                "Q3": {
                    "file_loc": "./content/companies/INTC/2022/10Q/Q3/INTC_2022_10Q_Q3.pdf",
                    "index_loc": "./content/companies/INTC/2022/10Q/Q3/index"
                },
            },
            "10K":  {
                "file_loc": "./content/companies/INTC/2022/10K/INTC_2022_10K.pdf",
                "index_loc": "./content/companies/INTC/2022/10K/index"
            },
            "other": {
                "report_type": "<report_type>",
                "file_loc": "<file_loc>",
                "index_loc": "<index_loc",
            }, 
        },
        "2021": {
            "10Q": {
                "Q1": {
                    "file_loc": "./content/companies/INTC/2021/10Q/Q1/INTC_2021_10Q_Q1.pdf",
                    "index_loc": "./content/companies/INTC/2021/10Q/Q1/index"
                },
                "Q2": {
                    "file_loc": "./content/companies/INTC/2021/10Q/Q2/INTC_2021_10Q_Q2.pdf",
                    "index_loc": "./content/companies/INTC/2021/10Q/Q2/index"
                },
                "Q3": {
                    "file_loc": "./content/companies/INTC/2021/10Q/Q3/INTC_2021_10Q_Q3.pdf",
                    "index_loc": "./content/companies/INTC/2021/10Q/Q3/index"
                },
            },
            "10K":  {
                "file_loc": "./content/companies/INTC/2021/10K/INTC_2021_10K.pdf",
                "index_loc": "./content/companies/INTC/2021/10K/index"
            },
            "other": {
                "file_loc": "<file_loc>",
                "index_loc": "<index_loc",
            }, 
        }
    }
} 
"""        
    
class FileManager:
    def __init__(self, path, index_name=None):
        self.path = path
        if index_name == None:
            self.index_name = "index"
        else:
            self.index_name = index_name.replace("/", "_")


    # gets companies using predefined directory structure
    def load(self):
        self.dir_dict = self._create_dict(self.path)

    def move_file(self, src_file, company_name, year, report_type, quarter=None):
        # Construct the new directory path
        new_dir = os.path.join(self.path, company_name, str(year), report_type)
        if quarter:
            new_dir = os.path.join(new_dir, quarter)

        # Create the new directory if it doesn't exist
        os.makedirs(new_dir, exist_ok=True)

        # Construct the new file name
        new_file_name = f"{company_name}_{year}_{report_type}"
        if quarter:
            new_file_name += f"_{quarter}"
        new_file_name += ".pdf"

        # Delete any existing PDF file in the directory
        for file in glob.glob(os.path.join(new_dir, "*.pdf")):
            os.remove(file)

        # Construct the new file path
        new_file_path = os.path.join(new_dir, new_file_name)

        # Move and rename the file
        shutil.move(src_file, new_file_path)
        self.load()
        return new_file_path

    def _create_dict(self, path):
        dir_dict = {}
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                if item == 'index':
                    # Check if there is a file in the directory that includes self.index_name in its file_name
                    if any(self.index_name in filename for filename in os.listdir(item_path)):
                        dir_dict['index_loc'] = item_path
                else:
                    dir_dict[item] = self._create_dict(item_path)
            elif item.endswith('.pdf'):
                dir_dict['file_loc'] = item_path
        return dir_dict
    
    def get_companies(self):
        assert isinstance(self.dir_dict, dict)
        return list(self.dir_dict.keys())

    def get_years(self, companies):
        if companies:
            return sorted(set.intersection(*(set(self.dir_dict[company].keys()) for company in companies)))
        return []

    def get_report_types(self, companies, years):
        if companies and years:
            return sorted(set.intersection(*(set(self.dir_dict[company][year].keys()) for company in companies for year in years)))
        return []

    def get_quarters(self, companies, years, report_types):
        keys = [set(self.dir_dict[company][year][report_type].keys()) for company in companies for year in years for report_type in report_types if report_type == '10Q' and report_type in self.dir_dict[company][year]]
        if keys:
            return sorted(set.intersection(*keys))
        else:
            return []

    def get_file_path(self, company, year, report_type, quarter=None):
        if quarter is not None:
            search_path = os.path.join(self.path, company, year, report_type, quarter)
        else:
            search_path = os.path.join(self.path, company, year, report_type)

        for root, dirs, files in os.walk(search_path):
            for file in files:
                if file.endswith(".pdf"):
                    return os.path.join(root, file)

        return None

    def create_index(self, company, year, report_type, quarter=None):
        dir_path = None
        if quarter!=None:
            dir_path = os.path.join(self.path, company, year, report_type, quarter, "index")
        else:
            dir_path = os.path.join(self.path, company, year, report_type, "index")
        if os.path.isdir(dir_path):
            return dir_path
        else:
            return None

    def index_exists(self, company, year, report_type, quarter=None):
        if quarter != None:
            if self.dir_dict[company][year][report_type][quarter]["index_loc"] != None:
                # print("file_manager - index_exists: true")
                return True
        else:
            if self.dir_dict[company][year][report_type]["index_loc"] != None:
                # print("file_manager - index_exists: true")
                return True
        # print("file_manager - index_exists: false")
        return False

    # create embeddings for all companies, years in directory structure
    # if company is specified, generate word embeddings only for that company
    # if year is specified along with company, generate word embeddings only for that company and year
    def generate_embeddings(self, index_gen, sp_company=None, sp_year=None):
        # do a specific company if applicable
        dirs_to_gen = []
        if sp_company:
            if sp_company in self.companies:
                comp = self.companies[sp_company]

                # do a specific year if applicable
                if sp_year:
                    if sp_year in comp:
                        directory = os.path.join(self.path, sp_company, sp_year)
                        index_path = os.path.join(directory, "index")
                        dirs_to_gen.append({"dir": directory, "index": index_path, "company": sp_company, "year": sp_year})
                    else:
                        # print("Invalid year.")
                        pass
                # else do for all years
                else:
                    for year in comp.keys():
                        directory = os.path.join(self.path, sp_company, year)
                        index_path = os.path.join(directory, "index")
                        dirs_to_gen.append({"dir": directory, "index": index_path, "company": sp_company, "year": year})
                        
            
            else:
                # print("Invalid company ticker.")
                pass

        else:
            for company in self.companies.keys():
                for year in self.companies[company].keys():
                    directory = os.path.join(self.path, company, year)
                    index_path = os.path.join(directory, "index")
                    dirs_to_gen.append({"dir": directory, "index": index_path, "company": company, "year": year})

        # generate embeddings now
        for dir_index in dirs_to_gen:
            # print(f"Generating {dir_index['company']} {dir_index['year']} vector store...")
            vector_store = index_gen.generate_vector_store_pdf_dir(dir_index['dir'])
            index_gen.save_vector_store(vector_store, dir_index["index"])

                

                    
""" Session Manager
manages the sessions created by the user.
This includes loading up previous sessions, 
saving them, and maintaining them """
class SessionManager:

    def __init__(self, save_file=None, _session_cls=None):
        self.sessions = None
        self.initialized = False
        self.active_session = None
        self.save_file = save_file
        self._session_cls = _session_cls
    
    def load(self):
        ss_list = {}
        # print("loading sessions...")
        self.sessions = {}
        try:
            with open(self.save_file, "r") as json_file:
                ss_list = json.load(json_file) 
            for name, ss in ss_list.items():
                session = self._session_cls.from_dict(ss)
                self.sessions[name] = session
        except Exception as e:
            print(e)
        # print("done loading sessions: ", self.sessions)

        self.initialized = True

    def save(self):
        ss_list = {}
        for name, session in self.sessions.items():
            ss_list[name] = session.to_dict()
        with open(self.save_file, "w") as json_file:
            json.dump(ss_list, json_file) 
    
    def add_session(self, session):

        self.sessions[session.name] = session
    
    def set_active_session(self, session):
        self.active_session = session