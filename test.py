from util import FileManager

def dict_to_list( d, parent_keys=[]):
    lst = []
    for k, v in d.items():
        new_keys = parent_keys + [k]
        if isinstance(v, dict):
            lst.extend(dict_to_list(v, new_keys))
        else:
            lst.append({"file_loc": v, "index_loc": "_".join(new_keys)})
    return lst

fm = FileManager('./content/companies')
fm.load()

ll = dict_to_list(fm.dir_dict)
for l in ll:
    print(l)
