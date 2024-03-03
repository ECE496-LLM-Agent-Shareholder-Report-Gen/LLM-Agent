from model_loader import ModelLoader
from retrievers import BasicRetriever, MultiCompanyYearRetriever
from pre_processing import IndexGenerator
from util import CompanyDirectory, InDirectory
import langchain
from langchain.memory import ConversationBufferMemory

#test commit

""" Session.
An instance of a conversation between user and chatbot.
Holds the companies, and years user can ask questions on.
Holds memory of conversation.
"""
class Session():
    
    

    def __init__(self, name, cd, llm, embeddings, cds_path, mode='basic'):
        self.name = name
        self.companies = cd.get_companies_from_user()
        possible_years = cd.get_years_intersection(self.companies)
        self.years = cd.get_years_for_companies(possible_years)
        self.questions_asked = 0
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        self.mode = mode

        if mode == 'multi':
            self.retriever = MultiCompanyYearRetriever(self.companies, self.years,cds_path, llm, embeddings)
        else:
            self.retriever = BasicRetriever(llm, self.index, embeddings)

    def answer(self, query, *args):
        self.questions_asked += 1
        self.retriever.answer(query)
        print("")



""" Application
holds the application state. """
class Application():
    no_session_text = "Not currently in an active session."
    default_prompt = "LLM Agent >> "
    
#model args add which model to use
    def __init__(self, model_args="llama-2-13b-chat", embeddings_args=None):
        # create the language models
        self.model_loader = ModelLoader(model_args)
        self.index_gen = IndexGenerator()

        # create the company directory
        self.base_path = "/groups/acmogrp/Large-Language-Model-Agent/"

        self.cds_path = "/groups/acmogrp/Large-Language-Model-Agent/content/companies"
        self.in_path = "/groups/acmogrp/Large-Language-Model-Agent/content"
        self.company_directory = CompanyDirectory(self.cds_path)
        self.in_directory = InDirectory(self.in_path, self.cds_path, [".pdf"], self.index_gen)
        self.prompt = self.default_prompt
        self.active_session = None
        self.sessions = {}

    # start the application
    def start(self):
        welcome_text = """Welcome to the LLM Agent Chat bot! 
        To learn a list of commands, please type !help"""
        print(welcome_text)

        while True:
            user_input = input(self.prompt)
            command = self.parse_input(user_input)
            if command["is_valid"]:
                if command and  command["is_cmd"]:
                    if command["cmd"] in self.available_commands:
                        if self.check_usage(command["cmd"], command["args"]):
                            self.available_commands[command["cmd"]]["func"](self, command["cmd"], command["args"])
                elif command and "query" in command:
                    self.answer(command["query"])


    # parses user input
    def parse_input(self, user_input):
        command = {}
        if not user_input.strip():
            command = {
                "is_valid": False,
                "is_cmd": False,
            }
            return command
        parts = user_input.split()
        if parts[0][0] == "!":
            command = {
                "is_valid": True,
                "is_cmd": True,
                "cmd": parts[0][1:].lower(),
                "args": parts[1:]
            }
        else:
            command = {
                "is_valid": True,
                "is_cmd": False,
                "query": user_input.strip()
            }

        return command


    # ====================================================
    # ================= Utility Methods ==================
    # ====================================================

    # decorator for checking for active session
    def session_active(func):
        # Define the wrapper function
        def wrapper(self, *args, **kwargs):
            if not self.active_session:
                print(self.no_session_text)
                return
            result = func(self, *args, **kwargs)
            return result
        # Return the wrapper function
        return wrapper
    

    # just checks number of args
    def check_usage(self, cmd, args):
        command_details = self.available_commands[cmd]
        if not "num_args" in command_details:
            return True
        elif not len(args) in command_details["num_args"] :
            print("Incorrect number of arguments provided!")
            print(f"Usage: {command_details['usage']}")
            return False
        return True
    
    # prints session details
    def print_session_details(self, ses):
        print(ses.name)
        print(f"\tConversation Length: {ses.questions_asked}")
        print(f"\tCompanies: {', '.join(ses.companies)}")
        print(f"\tYears: {', '.join(ses.years)}")


    # ====================================================
    # ================ Available Commands ================
    # ====================================================

    # Answer user's question
    # <user question...>
    @session_active
    def answer(self, query):
        self.active_session.answer(query)


    # update company directory structure
    def update_cd(self):
        self.company_directory.get_companies()


    # display list of available commands
    # !help [command]*
    def app_help(self, cmd, args):
        if not len(args):
            for key, value in self.available_commands.items():
                print(f"{key}")
                print(f"\t{value['help']}")
                print(f"\tUsage: {value['usage']}")
                if value["session_cmd"]:
                    print("\tThis is a session command.")
                print("")
        else:
            for arg in args:
                arg = arg.lower()
                if arg.lower() in self.available_commands:
                    print(f"{arg}")
                    print(f"\t{self.available_commands[arg]['help']}")
                    print(f"\tUsage: {self.available_commands[arg]['usage']}")
                    if self.available_commands[arg]["session_cmd"]:
                        print("\tThis is a session command.")
                    print("")
                else:
                    print(f"'{arg}' is not a valid command.")



    # exit application
    # !exit
    def app_exit(self, cmd, args):
        quit()


    # enter a session
    # !enter [session_name]
    def app_enter_session(self, cmd, args):
        session_name = args[0].lower()
        if session_name in self.sessions:
            self.active_session = self.sessions[session_name]
            self.prompt = f"{self.active_session.name} >> "


    # list sessions
    # !ls
    def app_list_sessions(self, cmd, args):
        if len(args) == 1 and args[0] == "-s":
            for session in self.sessions.values():
                self.print_session_details(session)
        else:
            for session_name in self.sessions.keys():
                print(session_name)

    # create and enter a session
    # !create [session_name]
    def app_create_session(self, cmd, args):
        self.company_directory.print_companies_all()
        session_name = args[0].lower()
        session = Session(args[0], self.company_directory, self.model_loader.language_model, self.index_gen.embeddings, self.cds_path, mode="multi")
        self.sessions[session_name] = session
        self.app_enter_session("enter", [args[0].lower()])


    # check if there are any new files
    # !in
    def app_check_files(self, cmd, args):
        files = self.in_directory.check_files()
        if files:
            print("The following files were found:")
            for file in files:
                print(f"\t{file}")
        else:
            print("No files found.")



    # check if there are any new files
    # !lc
    def app_check_companies(self, cmd, args):
        self.company_directory.get_companies()
        self.company_directory.print_companies_all()
       


    # load files in the in directory, and merges file embeddings to existing vector store,
    # or creates one from scratch if it does not exist
    # !load
    def app_load_files(self, cmd, args):
        self.in_directory.load_files(self.company_directory)
        self.company_directory.get_companies()

    
    # generate embeddings
    # !gen
    def app_gen_index(self, cmd, args):
        kwargs = {}
        if len(args) >= 1:
            kwargs["sp_company"] = args[0].upper()
        if len(args) >= 2:
            kwargs["sp_year"] = args[1]

        self.company_directory.generate_embeddings(self.index_gen, **kwargs)


    # lists all available session commands.
    # !shelp
    def ses_help(self, cmd, args):
        if not len(args):
            for key, value in self.available_commands.items():
                if value["session_cmd"]:
                    print(f"{key}")
                    print(f"\t{value['help']}")
                    print(f"\tUsage: {value['usage']}")
                    print("")
        else:
            for arg in args:
                arg = arg.lower()
                if arg in self.available_commands and self.available_commands[arg]["session_cmd"]:
                    print(f"{arg}")
                    print(f"\t{self.available_commands[arg]['help']}")
                    print(f"\tUsage: {self.available_commands[arg]['usage']}")
                    print("")
                elif not arg in self.available_commands:
                    print(f"'{arg}' is not a valid command.\n")
                elif not self.available_commands[arg]["session_cmd"]:
                    print(f"'{arg}' is not a session command.\n")

    
    # exit current session
    # !exit
    @session_active
    def ses_exit(self, cmd, args):
        self.active_session = False
        self.prompt = self.default_prompt


    # show session details
    # !sd
    @session_active
    def ses_details(self, cmd, *args):
        ses = self.active_session
        self.print_session_details(ses)

    # TODO: Add commands that allow user to ask questions on temporary reports
        # no need to to save the reports in the ./content/companies directory



    available_commands = {
        "help": {
            "func": app_help,
            "usage": "!help [command]*",
            "help": "Shows a list of all available commands, or gives more info on a specific command",
            "session_cmd": False,
        },
        "exit": {
            "func": app_exit,
            "usage": "!exit",
            "help": "Exits the application.",
            "session_cmd": False,
        },
        "enter":{
            "func": app_enter_session,
            "usage": "!enter [session_name]",
            "help": "Enter a session.",
            "session_cmd": False,
            "num_args": [1]
        },
        "create":{
            "func": app_create_session,
            "usage": "!create [session_name]",
            "help": "Create and enter a session.",
            "session_cmd": False,
            "num_args": [1]
        },
        "ls":{
            "func": app_list_sessions,
            "usage": "!ls [-s]",
            "help": "List available sessions. Add the '-s' option for session details",
            "session_cmd": False,
            "num_args": [0, 1]
        },
        "in":{
            "func": app_check_files,
            "usage": "!in",
            "help": "Check for any files in the 'IN' directory",
            "session_cmd": False,
        },
        "lc":{
            "func": app_check_companies,
            "usage": "!lc",
            "help": "Show all the companies that can be found in the Company Directory Structure.",
            "session_cmd": False,
        },
        "load":{
            "func": app_load_files,
            "usage": "!load",
            "help": "Loads any files in the 'IN' directory into the Company Directory Structure. This needs to be run before generating any embeddings.",
            "session_cmd": False,
        },
        "gen":{
            "func": app_gen_index,
            "usage": "!gen [company [year]]",
            "help": "Generate the vector store for the company/companies. Specify a company and year for more specific (and faster) vector store creation.",
            "session_cmd": False,
            "num_args": [0, 1, 2]
        },
        "shelp": {
            "func": ses_help,
            "usage": "!shelp [command]*",
            "help": "Shows a list of all available session commands, or gives more info on a specific session command.",
            "session_cmd": True,
        },
        "sexit": {
            "func": ses_exit,
            "usage": "!sexit",
            "help": "Exits the session.",
            "session_cmd": True,
        },
        "sd": {
            "func": ses_details,
            "usage": "!sd",
            "help": "Show the current session's details.",
            "session_cmd": True,
        },

    }



""" main.
Entry into application
"""
def main():
    print("Please wait a moment while we get ready...")
    app = Application()

    app.start()



if __name__ == '__main__':
    main()