import os
import json
import glob
import shutil


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
    """
    A utility class for managing directories, files, and indexing within a specified path.

    Args:
        path (str): The base directory path.
        index_name (str, optional): The name for the index directory (default is "index").

    Attributes:
        path (str): The base directory path.
        index_name (str): The name for the index directory.

    Methods:
        - create_directory(directory_path: str) -> None:
            Creates a directory if it doesn't exist.

        - load() -> None:
            Loads the directory structure into memory.

        - move_file(src_file: str, company_name: str, year: int, report_type: str, quarter: str = None) -> str:
            Moves and renames a file to a new location based on company, year, and report type.

        - get_companies() -> List[str]:
            Returns a list of company names available in the directory structure.

        - get_years(companies: List[str]) -> List[int]:
            Returns a sorted list of years available for the specified companies.

        - get_report_types(companies: List[str], years: List[int]) -> List[str]:
            Returns a sorted list of report types available for the specified companies and years.

        - get_quarters(companies: List[str], years: List[int], report_types: List[str]) -> List[str]:
            Returns a sorted list of quarters available for the specified companies, years, and report types.

        - get_file_path(company: str, year: int, report_type: str, quarter: str = None) -> Optional[str]:
            Returns the file path for a specific company, year, report type, and optional quarter.

        - create_index(company: str, year: int, report_type: str, quarter: str = None) -> Optional[str]:
            Creates an index directory path for a specific company, year, report type, and optional quarter.

        - index_exists(company: str, year: int, report_type: str, quarter: str = None) -> bool:
            Checks if an index exists for a specific company, year, report type, and optional quarter.

        - generate_embeddings(index_gen, sp_company: str = None, sp_year: int = None) -> None:
            Generates word embeddings for specified companies and years using an index generator.
    """
    def __init__(self, path, index_name=None):
        """
        Initializes a FileManager instance.

        Args:
            path (str): The base directory path.
            index_name (str, optional): The name of the index directory. Defaults to "index".
        """
        self.path = path
        if index_name == None:
            self.index_name = "index"
        else:
            self.index_name = index_name.replace("/", "_")
        self.create_directory(path)
        
    def create_directory(self, directory_path):
        """
        Creates a directory if it does not exist.

        Args:
            directory_path (str): The path of the directory to create.
        """
        # Check if the directory exists
        if not os.path.exists(directory_path):
            # Create the directory
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' created successfully.")
        else:
            print(f"Directory '{directory_path}' already exists.")

    # Example usage: create a directory named "my_directory"


    # gets companies using predefined directory structure
    def load(self):
        """
        Loads the directory structure into memory.
        """
        self.dir_dict = self._create_dict(self.path)

    def move_file(self, src_file, company_name, year, report_type, quarter=None):
        """
        Moves a file to the specified directory and renames it.

        Args:
            src_file (str): The source file path.
            company_name (str): The company name.
            year (int): The year.
            report_type (str): The report type.
            quarter (str): The quarter (optional). Defaults to None.

        Returns:
            str: The new file path.
        """
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
        """
        Recursively creates a dictionary representing the directory structure.

        Args:
            path (str): The path to the root directory.

        Returns:
            dict: A dictionary representing the directory structure.
        """
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
        """
        Returns a list of company names available in the directory structure.

        Returns:
            list: A list of company names.
        """
        assert isinstance(self.dir_dict, dict)
        return list(self.dir_dict.keys())

    def get_years(self, companies):
        """
        Returns a list of years available for the specified companies.

        Args:
            companies (list): A list of company names.

        Returns:
            list: A list of years.
        """
        if companies:
            return sorted(set.intersection(*(set(self.dir_dict[company].keys()) for company in companies)))
        return []

    def get_report_types(self, companies, years):
        """
        Returns a list of report types available for the specified companies and years.

        Args:
            companies (list): A list of company names.
            years (list): A list of years.

        Returns:
            list: A list of report types.
        """
        if companies and years:
            return sorted(set.intersection(*(set(self.dir_dict[company][year].keys()) for company in companies for year in years)))
        return []

    def get_quarters(self, companies, years, report_types):
        """
        Returns a list of quarters available for the specified companies, years, and report types.

        Args:
            companies (list): A list of company names.
            years (list): A list of years.
            report_types (list): A list of report types.

        Returns:
            list: A list of quarters.
        """
        keys = [set(self.dir_dict[company][year][report_type].keys()) for company in companies for year in years for report_type in report_types if report_type == '10Q' and report_type in self.dir_dict[company][year]]
        if keys:
            return sorted(set.intersection(*keys))
        else:
            return []

    def get_file_path(self, company, year, report_type, quarter=None):
        """
        Returns the file path for the specified company, year, report type, and quarter (if provided).

        Args:
            company (str): The company name.
            year (str): The year.
            report_type (str): The report type (e.g., '10K', '10Q').
            quarter (str, optional): The quarter (e.g., 'Q1', 'Q2'). Defaults to None.

        Returns:
            str: The file path or None if not found.
        """
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
        """
        Creates an index directory for the specified company, year, report type, and quarter (if provided).

        Args:
            company (str): The company name.
            year (str): The year.
            report_type (str): The report type (e.g., '10K', '10Q').
            quarter (str, optional): The quarter (e.g., 'Q1', 'Q2'). Defaults to None.

        Returns:
            str: The index directory path or None if not found.
        """
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
        """
        Checks if an index exists for the specified company, year, report type, and quarter (if provided).

        Args:
            company (str): The company name.
            year (str): The year.
            report_type (str): The report type (e.g., '10K', '10Q').
            quarter (str, optional): The quarter (e.g., 'Q1', 'Q2'). Defaults to None.

        Returns:
            bool: True if the index exists, False otherwise.
        """
        if quarter != None:
            if "index_loc" in self.dir_dict[company][year][report_type][quarter]:
                # print("file_manager - index_exists: true")
                return True
        else:
            if "index_loc" in self.dir_dict[company][year][report_type]:
                # print("file_manager - index_exists: true")
                return True
        # print("file_manager - index_exists: false")
        return False

    def generate_embeddings(self, index_gen, sp_company=None, sp_year=None):
        """
        Generates word embeddings for all companies and years in the directory structure.

        Args:
            index_gen: The index generator object.
            sp_company (str, optional): The specific company name. Defaults to None.
            sp_year (str, optional): The specific year. Defaults to None.
        """
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

                

                    
class SessionManager:
    """
    Manages sessions and provides methods for loading, saving, adding, and setting active sessions.

    Args:
        save_file (str, optional): Path to the file where session data will be saved. Defaults to None.
        _session_cls (class, optional): Class representing individual sessions. Defaults to None.

    Attributes:
        sessions (dict): A dictionary to store session objects.
        initialized (bool): Indicates whether the manager has been initialized.
        active_session: The currently active session.
        save_file (str): Path to the file where session data will be saved.
        _session_cls (class): Class representing individual sessions.

    Methods:
        - load(): Loads session data from the specified file.
        - save(): Saves session data to the specified file.
        - add_session(session): Adds a session to the manager.
        - set_active_session(session): Sets the active session.

    Example usage:
        session_manager = SessionManager(save_file="sessions.json", _session_cls=MySession)
        session_manager.load()
        session_manager.add_session(my_session)
        session_manager.set_active_session(my_session)
        session_manager.save()
    """
    def __init__(self, save_file=None, _session_cls=None):
        self.sessions = None
        self.initialized = False
        self.active_session = None
        self.save_file = save_file
        self._session_cls = _session_cls
    
    def load(self):
        """
        Loads session data from the specified file.

        Raises:
            Exception: If there is an error during loading.

        Example usage:
            session_manager.load()
        """
        ss_list = {}
        self.sessions = {}
        try:
            with open(self.save_file, "r") as json_file:
                ss_list = json.load(json_file) 
            for name, ss in ss_list.items():
                session = self._session_cls.from_dict(ss)
                self.sessions[name] = session
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(e)

        self.initialized = True

    def save(self):
        """
        Saves session data to the specified file.

        Example usage:
            session_manager.save()
        """
        ss_list = {}
        os.makedirs(os.path.dirname(self.save_file), exist_ok=True)
        for name, session in self.sessions.items():
            ss_list[name] = session.to_dict()
        with open(self.save_file, "w") as json_file:
            json.dump(ss_list, json_file) 
    
    def add_session(self, session):
        """
        Adds a session to the manager.

        Args:
            session: An instance of the session class.

        Example usage:
            session_manager.add_session(my_session)
        """
        self.sessions[session.name] = session
    
    def set_active_session(self, session):
        """
        Sets the active session.

        Args:
            session: An instance of the session class.

        Example usage:
            session_manager.set_active_session(my_session)
        """
        self.active_session = session