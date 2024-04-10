from abc import ABC, abstractmethod


class TemplateFormatter(ABC):
    """
    Abstract base class for template formatters.

    Attributes:
        None
    """

    @abstractmethod
    def init_template(self, system_message, instruction):
        """
        Initialize a template.

        Args:
            system_message (str): The system message.
            instruction (str): The instruction.

        Returns:
            str: The initialized template.
        """
        pass

    @abstractmethod
    def init_template_from_memory(self, system_message, instruction, memory):
        """
        Initialize a template from memory.

        Args:
            system_message (str): The system message.
            instruction (str): The instruction.
            memory (list): List of memory instances.

        Returns:
            str: The initialized template.
        """
        pass

    @abstractmethod
    def add_instruction(self, template, instruction):
        """
        Add an instruction to the template.

        Args:
            template (str): The existing template.
            instruction (str): The instruction to add.

        Returns:
            str: The updated template.
        """
        pass

    @abstractmethod
    def add_response(self, template, response):
        """
        Add a response to the template.

        Args:
            template (str): The existing template.
            response (str): The response to add.

        Returns:
            str: The updated template.
        """
        pass


class NoTemplateFormatter(ABC):
    """
    Template formatter that does not use templates.

    Attributes:
        None
    """

    def init_template(self, system_message, instruction):
        """
        Initialize a template.

        Args:
            system_message (str): The system message.
            instruction (str): The instruction.

        Returns:
            str: The combined system message and instruction.
        """
        return system_message +"\n\n"+ instruction

    def init_template_from_memory(self, system_message, instruction, memory):
        """
        Initialize a template from memory.

        Args:
            system_message (str): The system message.
            instruction (str): The instruction.
            memory (list): List of memory instances.

        Returns:
            str: The combined system message, instruction, and memory.
        """
        return system_message +"\n\n"+ instruction + "\n\n" + memory

    def add_instruction(self, template, instruction):
        """
        Add an instruction to the template (no effect).

        Args:
            template (str): The existing template.
            instruction (str): The instruction to add.

        Returns:
            str: The unchanged template.
        """
        pass

    def add_response(self, template, response):
        """
        Add a response to the template (no effect).

        Args:
            template (str): The existing template.
            response (str): The response to add.

        Returns:
            str: The unchanged template.
        """
        pass

class LlamaTemplateFormatter(TemplateFormatter):
    """
    Template formatter using Llama-style templates.

    Attributes:
        sys_beg (str): Beginning of system message.
        sys_end (str): End of system message.
        ai_n_beg (str): Beginning of AI-generated response.
        ai_n_end (str): End of AI-generated response.
        usr_n_beg (str): Beginning of user instruction.
        usr_n_end (str): End of user instruction.
        usr_0_beg (str): Beginning of user memory instance.
        usr_0_end (str): End of user memory instance.
    """

    sys_beg = "<s>[INST] <<SYS>>\n"
    sys_end = "\n<</SYS>>\n\n"
    ai_n_beg = " "
    ai_n_end = " </s>"
    usr_n_beg = "<s>[INST] "
    usr_n_end = " [/INST]"
    usr_0_beg = ""
    usr_0_end = " [/INST]"

    def init_template(self, system_message, instruction):
        """
        Initialize a Llama-style template.

        Args:
            system_message (str): The system message.
            instruction (str): The instruction.

        Returns:
            str: The initialized Llama-style template.
        """
        template = f"{self.sys_beg}{system_message}{self.sys_end}"
        template += f"{self.usr_0_beg}{instruction}{self.usr_0_end}"
        return template

    def init_template_from_memory(self, system_message, instruction, memory):
        """
        Initialize a Llama-style template from memory.

        Args:
            system_message (str): The system message.
            instruction (str): The instruction.
            memory (list): List of memory instances.

        Returns:
            str: The initialized Llama-style template.
        """
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

    