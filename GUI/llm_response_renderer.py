import streamlit as st

import os

"""Everything pertaining to the sessions for the GUI. 
This class creates the session itself, and is responsible for
populating Sessions."""
class LLMResponseRenderer:

    def __init__(self, session, llm_chain):
        self.session = session
        self.llm_chain = llm_chain
    
    def render(self, question):
         # Add user message to chat history
        if self.session.llm_chain == "Simple Chain":
            self.render_simple_response(question)
        elif self.session.llm_chain == "Fusion Chain":
            self.render_fusion_response(question)
        elif self.session.llm_chain == "Stepback Chain":
            self.render_stepback_response(question)
        elif self.session.llm_chain == "Simple Stepback Chain":
            self.render_simple_stepback_response(question)

    """ Render simple llm chain response """
    def render_simple_response(self, question):
        stream = self.session.chatbot.stream(question)
        response = st.write_stream(stream)
        return response

    """ Render fusion llm chain response """
    def render_fusion_response(self, question):
        all_responses = []

        # generate sub queries
        sub_query_stream = self.session.chatbot.stream_sub_query_response(question)

        sub_query_response = st.write_stream(sub_query_stream)
        all_responses.append(sub_query_response)

        # check context
        context = self.session.chatbot.valid_context(sub_query_response)
        
        if context == None:
            all_responses.append("Failed to retrieve context, we might not have been able to parse the LLM's sub queries, pleast try again.")
        else:
            final_stream = self.session.chatbot.stream_result_response(question, context)
            final_stream_response = st.write_stream(final_stream)
            all_responses.append(final_stream_response)

        full_response = "\n\n".join(all_responses)

        return full_response

    
    """ Render stepback llm chain response """
    def render_stepback_response(self, question):
        all_responses = []

        # generate sub queries
        sub_query_gen_stream = self.session.chatbot.stream_sub_query_gen(question)
        sub_query_response = st.write_stream(sub_query_gen_stream)

        all_responses.append(sub_query_response)

        # answer each sub query
        sub_query_responses_streams_question = self.session.chatbot.stream_sub_query_responses(sub_query_response)

        sub_queries = [q for s, q in sub_query_responses_streams_question]
        sub_query_answers = []

        for stream, sub_query in sub_query_responses_streams_question:
            st.write(f"<b>{sub_query}:</b>", unsafe_allow_html=True)
            sub_query_answer = st.write_stream(stream)
            sub_query_answers.append(sub_query_answer)
            all_responses.append(f"{sub_query}\n{sub_query_answer}")

        # get final result
        final_stream = self.session.chatbot.stream_final_response(question, sub_queries, sub_query_answers)
        final_response = st.write_stream(final_stream)

        all_responses.append(final_response)

        full_response = "\n\n".join(all_responses)

        return full_response

    """ Render simple stepback llm chain response """
    def render_simple_stepback_response(self, question):
        stream = self.session.chatbot.stream(question)
        response = st.write_stream(stream)
        return response
