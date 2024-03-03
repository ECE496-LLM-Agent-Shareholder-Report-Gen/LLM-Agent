import os
from pathlib import Path
import shutil




""" Company Directory
manages the company directory structure of:
        {path}/company/year
"""        
    
class CompanyDirectory:
    def __init__(self, path):
        self.path = path
        self.companies = {}
        self.get_companies()

    # gets companies using predefined directory structure
    def get_companies(self):
        # all directories under {path}
        for company in os.listdir(self.path):
    
            # only take directories, no files
            if os.path.isdir(os.path.join(self.path, company)):
                # list of tuples (year, year_path)
                self.companies[company] = {}

                # look for all the years corresponding to this copmany
                for year in os.listdir(os.path.join(self.path, company)):
    
                    # make sure it is a directory
                    if os.path.isdir(os.path.join(self.path, company, year)):
                        self.companies[company][year] = {}
                        # make sure it has atleast 1 file
                        # comps = {
                        #     "AMD": {
                        #         "2022": {
                        #             "10Q": {
                        #                 "Q1": "./content/companies/AMD/2022/AMD_2022_10Q_Q1.pdf",
                        #                 "Q2": "./content/companies/AMD/2022/AMD_2022_10Q_Q2.pdf",
                        #                 "Q3": "./content/companies/AMD/2022/AMD_2022_10Q_Q3.pdf"
                        #             },
                        #             "10K":  "./content/companies/AMD/2022/AMD_2022_10K.pdf",
                        #             "index":  "./content/companies/AMD/2022/index"
                        #         }
                        #     }
                        # }
                        for pdf_or_index in os.listdir(os.path.join(self.path, company, year)):
                            # check if index
                            if pdf_or_index == 'index' and os.path.isdir(os.path.join(self.path, company, year, pdf_or_index)):
                                self.companies[company][year][pdf_or_index] = os.path.join(self.path, company, year, pdf_or_index)

                            # do the files
                            if os.path.isfile(os.path.join(self.path, company, year, pdf_or_index)):
                                # check if 10Q
                                if pdf_or_index.find("10Q") != -1:
                                    # create dict if not already exisiting
                                    if not "10Q" in self.companies[company][year]:
                                        self.companies[company][year]["10Q"] = {}
                                    if pdf_or_index.find("Q1") != -1:
                                        self.companies[company][year]["10Q"]["Q1"] = pdf_or_index
                                    elif pdf_or_index.find("Q2") != -1:
                                        self.companies[company][year]["10Q"]["Q2"] = pdf_or_index
                                    elif pdf_or_index.find("Q3") != -1:
                                        self.companies[company][year]["10Q"]["Q3"] = pdf_or_index
                                elif pdf_or_index.find("10K") != -1:
                                    self.companies[company][year]["10K"] = pdf_or_index

    # print all the companies available along with years and report types
    def print_companies_all(self):
        for company in sorted(self.companies.keys()):
            if len(self.companies[company].keys()) > 0:
                print(f"{company}:")
                for year in sorted(self.companies[company].keys()):
                    report_types = []
                    if "10Q" in self.companies[company][year]:
                        for Q_type in self.companies[company][year]["10Q"].keys():
                            report_types.append(Q_type)
                    if "10K" in self.companies[company][year]:
                        report_types.append("10K")
                    report_types.sort()
                    print(f"  {year}: {', '.join(report_types)}")
    
    
    # get years for the company
    def get_years_for_company(self, company):
        company = company.upper()
        if company in self.companies:
            return sorted([year[0] for year in self.companies[company]], reverse=True)
        else:
            return []


    # get possible years with the given companies
    def get_years_intersection(self, companies):
         # list of sets
        years = []
        for company in companies:
            print(company)
            if company in self.companies:
                years.append(set(self.companies[company]))

        year_set = set()
        for year_obj in years:
            temp = set([year for year in year_obj])
            if not year_set:
                year_set = temp
            else:
                year_set = temp & year_set
        intersection = sorted(list(year_set), reverse=True)
        print(f"Possible years for {', '.join(companies)} are {', '.join(intersection)}")
        return intersection


    # get years user wants to evaluate on given the available years 
    def get_years_for_companies(self, years):
       
        retries = 3
        while retries:
            start_year = input("Enter a start year: ")
            end_year = input("Enter an end year: ")
            if start_year.isdigit() and end_year.isdigit():
                start_year = int(start_year)
                end_year = int(end_year)
                print("start year: ", start_year)
                print("end year: ", end_year)
                if start_year <= end_year and str(start_year) in years and str(end_year) in years:
                    return [str(year) for year in range(start_year, end_year+1)]
                else:
                    print("somehow, not found: ", years)
            print("Invalid input. Please try again.")
            retries -= 1
        
        if retries == 0:
            print("We were unable to process your input, please try again.")
            return None


    # this method works
    # allows user to select the companies they want 
    def get_companies_from_user(self):
        companies = []
        while True:
            company = input("Enter a company name (or 'done' to finish): ")
            if company.lower() == 'done':
                break
            elif company.upper() in self.companies:
                companies.append(company.upper())
            else:
                print(f"Company '{company}' not found.")
        return companies


    def get_index(self, company, year):
        return os.path.join(self.path, company, year, "index")

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
                        print("Invalid year.")
                # else do for all years
                else:
                    for year in comp.keys():
                        directory = os.path.join(self.path, sp_company, year)
                        index_path = os.path.join(directory, "index")
                        dirs_to_gen.append({"dir": directory, "index": index_path, "company": sp_company, "year": year})
                        
            
            else:
                print("Invalid company ticker.")

        else:
            for company in self.companies.keys():
                for year in self.companies[company].keys():
                    directory = os.path.join(self.path, company, year)
                    index_path = os.path.join(directory, "index")
                    dirs_to_gen.append({"dir": directory, "index": index_path, "company": company, "year": year})

        # generate embeddings now
        for dir_index in dirs_to_gen:
            print(f"Generating {dir_index['company']} {dir_index['year']} vector store...")
            vector_store = index_gen.generate_vector_store_pdf_dir(dir_index['dir'])
            index_gen.save_vector_store(vector_store, dir_index["index"])

                



""" In Directory
checks for reports in the given directory
        {path}/IN
"""  
class InDirectory:
    def __init__(self, path, out_path, valid_extensions, index_generator):
        self.path = path
        self.in_path = os.path.join(path, "IN")
        self.out_path = out_path
        self.valid_extensions = valid_extensions
        self.index_generator = index_generator
    

    # checks for files in the IN directory
    def check_files(self):
        files = []
        for entry in os.scandir(self.in_path):
            if entry.is_file() and Path(entry.path).suffix in self.valid_extensions:
                files.append(entry.name)
        return files
    
    # loads all files
    def load_files(self, company_directory):
        for entry in os.scandir(self.in_path):
            if entry.is_file() and Path(entry.path).suffix in self.valid_extensions:
                print(f"File: {entry.name}")
                comp = ""
                year = ""
                report_type = ""
                quarter = ""

                # check for company
                while not comp:
                    comp = input("Company Ticker: ")
                    comp = comp.strip()
                comp = comp.upper()
                # check for year
                while not year:
                    year = input("Year: ")
                    try:
                        p = int(year)
                    except:
                        print("Invalid year")
                        year = ""

                # check for report type
                while not report_type:
                    report_type = input("Report type (10K, 10Q): ")
                    report_type = report_type.strip()
                    if report_type.upper() != "10K" and report_type.upper() != "10Q":
                        print("Invalid report type. Either 10K or 10Q")
                        report_type = ""
                    else:
                        report_type = report_type.upper()
                
                # check for quarter if applicable
                if report_type == "10Q":
                    while not quarter:
                        quarter = input("Quarter (1, 2, or 3): ")
                        quarter = quarter.strip()
                    if not quarter in ["1", "2", "3"]:
                        print("Invalid quarter. Either 1, 2, or 3.")
                        quarter = ""
                
                extensions = Path(entry.path).suffixes
                if quarter:
                    file_name = f"{comp}_{year}_{report_type}_Q{quarter}{''.join(extensions)}"
                else:
                    file_name = f"{comp}_{year}_{report_type}{''.join(extensions)}"
                dest_dir = os.path.join(self.out_path, comp, year)
                dest_file = os.path.join(dest_dir, file_name)

                os.makedirs(dest_dir, exist_ok=True) # create the new directory if it does not exist

                
                shutil.move(entry.path, dest_file)

                index_path = Path(os.path.join(dest_dir, "index"))
                
                if index_path.exists():
                    # check if user wants to update the existing vector store or recreate it incase of duplicates
                    print(f"Would you like to update the existing vector store at {index_path.as_posix()} (1) or recreate it (2)?")
                    update_or_create = input("1: update; 2: recreate >> ")
                    update_or_create = update_or_create.strip()
                    while update_or_create not in  ["1", "2"]:
                        print("Invalid input, either 1 or 2")
                        update_or_create = input("1: update; 2: recreate >> ")
                        update_or_create = update_or_create.strip()

                    # update the vector store
                    if update_or_create == "1":

                        print("Updating vector store at: ", index_path.as_posix())
                        vector_store1 = self.index_generator.load_vector_store(index_path.as_posix())
                        vector_store2 = self.index_generator.generate_vector_store_pdf_file(dest_file)
                        vector_store_merged = self.index_generator.merge_vector_stores(vector_store1, vector_store2)
                        self.index_generator.save_vector_store(vector_store_merged, index_path.as_posix())
                    
                    # recreate the vector store
                    elif update_or_create == "2":
                        company_directory.generate_embeddings(comp, year)

                # vector store does not exist, create it from scratch
                else:
                    print("Creating new vector store at: ", index_path.as_posix())
                    vector_store = self.index_generator.generate_vector_store_pdf_file(dest_file)
                    self.index_generator.save_vector_store(vector_store, index_path.as_posix())


                    

