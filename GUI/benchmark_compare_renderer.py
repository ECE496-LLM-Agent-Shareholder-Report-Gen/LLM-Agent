import json
import streamlit as st
import pandas as pd
import altair as alt
from GUI.misc import check_session_valid, render_session_info

class BenchmarkCompareRenderer:

    def __init__(self, global_singleton):
        self.global_singleton = global_singleton
        self.benchmark_dict = global_singleton.benchmark_session_manager.sessions
        if not "selected_sessions" in st.session_state:
            st.session_state["selected_sessions"] = []
    
    def render(self):
        self.render_header()
        self.render_select()
        st.divider()
        self.render_chart()

    def render_header(self):
        st.title("Compare your Benchmarks")


    def render_select(self):
        sessions = list(self.benchmark_dict.keys())
        sessions.sort()
        if len(sessions) > 0:
            st.session_state["selected_sessions"] = st.multiselect("Choose the benchmarks to compare", options=sessions)
        else:
            st.markdown("No benchmarks available to compare. Please click on 'Create New Benchmark' in the left-hand navigation bar to create a new benchmark.")

    def render_chart(self):
        st.subheader("Similarity Scores", divider="grey")

        qids = []
        data_raw = []
        column_names = []

        data_raw.append(qids)
        column_names.append("Q&A")
        for ses in st.session_state["selected_sessions"]:
            ses_scores = []
            column_names.append(ses)
            for idx, qae in self.benchmark_dict[ses].question_answer_expected.items():
                s_idx = str(idx)
                if s_idx not in qids:
                    qids.append(s_idx)
                ses_scores.append(qae.similarity_score)
            data_raw.append(ses_scores)


        if len(data_raw) > 1:
            df = pd.DataFrame(data_raw).T
            df.columns = column_names
            # Melt the DataFrame to long format
            data_melted = df.melt('Q&A', var_name='Benchmark', value_name='Similarity Score')
            # Create a grouped bar chart
            chart = alt.Chart(data_melted).mark_bar().encode(
                x=alt.X('Q&A:O', axis=alt.Axis(title='Q&A',labelAngle=0)),
                xOffset="Benchmark:N",
                y=alt.Y('Similarity Score:Q', axis=alt.Axis(title='Similarity Score')),
                color='Benchmark:N',
            )
            st.altair_chart(chart, use_container_width=True)
            # st.bar_chart(data=df, x="Q&A", y=actual_columns )
        else:
            st.markdown("No data to display")
        
