
from abc import ABC, abstractmethod


class TemplateFormatter(ABC):

    @abstractmethod
    def init_template(self, system_message, instruction):
        pass

    @abstractmethod
    def init_template_from_memory(self, system_message, instruction, memory):
        pass

    @abstractmethod
    def add_instruction(self, template, instruction):
        pass

    @abstractmethod
    def add_response(self, template, response):
        pass


class NoTemplateFormatter(ABC):

    def init_template(self, system_message, instruction):
        return system_message +"\n\n"+ instruction

    def init_template_from_memory(self, system_message, instruction, memory):
        return system_message +"\n\n"+ instruction + "\n\n" + memory
        pass

    def add_instruction(self, template, instruction):
        pass

    def add_response(self, template, response):
        pass

class LlamaTemplateFormatter(TemplateFormatter):

    sys_beg = "<s>[INST] <<SYS>>\n"
    sys_end = "\n<</SYS>>\n\n"
    ai_n_beg = " "
    ai_n_end = " </s>"
    usr_n_beg = "<s>[INST] "
    usr_n_end = " [/INST]"
    usr_0_beg = ""
    usr_0_end = " [/INST]"

    def init_template(self, system_message, instruction):
        template = f"{self.sys_beg}{system_message}{self.sys_end}"
        template += f"{self.usr_0_beg}{instruction}{self.usr_0_end}"
        return template

    def init_template_from_memory(self, system_message, instruction, memory):
        template = f"{self.sys_beg}{system_message}{self.sys_end}"

        # handle first memory instance
        if len(memory) > 0:
            template += f"{self.usr_0_beg}{memory[0][0]}{self.usr_0_end}"
            template = self.add_response(template, memory[0][1])

        # handle second+ memory instance
        if len(memory) > 1:
            for mem in memory[1:]:
                template = self.add_instruction(template, mem[0])
                template = self.add_response(template, mem[1])

        # append the instruction
        template = self.add_instruction(template, instruction)
        return template
    
    def add_instruction(self, template, instruction):
        template += f"{self.usr_n_beg}{instruction}{self.usr_n_end}"
        return template

    def add_response(self, template, response):
        template += f"{self.ai_n_beg}{response}{self.ai_n_end}"
        return template


class MixtralTemplateFormatter(TemplateFormatter):

    ai_n_beg = " "
    ai_n_end = " </s>"
    usr_n_beg = "<s>[INST] "
    usr_n_end = " [/INST]"

    def init_template(self, system_message, instruction):
        template = f"{self.usr_n_beg}{system_message}\n\n"
        template += f"{instruction}{self.usr_n_end}"
        return template

    def init_template_from_memory(self, system_message, instruction, memory):
        template = f"{self.usr_n_beg}{system_message}\n\n"

        # handle first memory instance
        if len(memory) > 0:
            template += f"{memory[0][0]}{self.usr_n_end}"
            template = self.add_response(template, memory[0][1])

        # handle second+ memory instance
        if len(memory) > 1:
            for mem in memory[1:]:
                template = self.add_instruction(template, mem[0])
                template = self.add_response(template, mem[1])

        # append the instruction
        if len(memory)>0:
            template+=f"{self.usr_n_beg}"
        template += f"{instruction}{self.usr_n_end}"
        return template
    
    def add_instruction(self, template, instruction):
        template += f"{self.usr_n_beg}{instruction}{self.usr_n_end}"
        return template

    def add_response(self, template, response):
        template += f"{self.ai_n_beg}{response}{self.ai_n_end}"
        return template

    