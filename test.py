# from util import CompanyDirectory

# cd = CompanyDirectory("../content/companies")
# cd.print_companies_all()
# comps = cd.get_companies_from_user()

# years = cd.get_years_intersection(comps)

from pathlib import Path
source = "AMD_2021_10K.pdf"
file_name = source.split(".")[0]
print(file_name)
if file_name.find("10Q") != -1:
    print(file_name.split("_"))