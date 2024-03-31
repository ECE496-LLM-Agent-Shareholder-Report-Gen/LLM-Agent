import streamlit as st
from streamlit_extras.stylable_container import stylable_container


def link_clicked(session, session_manager, is_chat_session=True):
    print("setting active session to ", session.name)
    if session_manager.active_session != None:
        session_manager.active_session.deinitialize()
    if is_chat_session:
        if not "switch_page_chat" in st.session_state:
            st.session_state["switch_page_chat"] = True
    else:
        if not "switch_page_bench" in st.session_state:
            st.session_state["switch_page_bench"] = True 
    session_manager.active_session = session


def navbar(global_singleton):
    # Sidebar
    with st.sidebar:
        st.subheader("Sessions", divider="grey")
        if global_singleton.chat_session_manager and global_singleton.chat_session_manager.sessions:
            for session_name, session in global_singleton.chat_session_manager.sessions.items():
                if session == global_singleton.chat_session_manager.active_session:
                     with stylable_container(
                        key="active_session",
                            css_styles="""
                                button {
                                    background-color: white;
                                    color: black;
                                }
                                """,
                        ):
                        st.button(session_name, 
                                            key=f"{session_name}_1",
                                            use_container_width=True)
                         
                else:
                    st.button(session_name, 
                                            key=f"{session_name}_2", 
                                            on_click=link_clicked, 
                                            args=[session, global_singleton.chat_session_manager], 
                                            use_container_width=True)
            if "switch_page_chat" in st.session_state and st.session_state["switch_page_chat"]:
                st.session_state["compare_active"] = False
                st.session_state["switch_page_chat"] = False
                if global_singleton.benchmark_session_manager.active_session:
                    global_singleton.benchmark_session_manager.active_session.deinitialize()
                st.switch_page("pages/chat_page.py")
            save_sessions = st.button("Save sessions", use_container_width=True)
            if save_sessions:
                global_singleton.chat_session_manager.save()
        else:
            st.markdown("No sessions")
        new_session = st.button("＋ Create new Session", use_container_width=True)
        if new_session:
            if global_singleton.benchmark_session_manager.active_session:
                    global_singleton.benchmark_session_manager.active_session.deinitialize()
                    global_singleton.benchmark_session_manager.active_session = None
            st.switch_page("pages/session_page.py")

        """ Benchmarks """
        st.subheader("Benchmarks", divider="grey")
        if global_singleton.benchmark_session_manager and global_singleton.benchmark_session_manager.sessions:
            # benchmarks
            for session_name, session in global_singleton.benchmark_session_manager.sessions.items():
                if session == global_singleton.benchmark_session_manager.active_session:
                     with stylable_container(
                        key="active_b_session",
                            css_styles="""
                                button {
                                    background-color: white;
                                    color: black;
                                }
                                """,
                        ):
                        st.button(session_name, 
                                            key=f"b_{session_name}_1", 
                                            use_container_width=True)
                         
                else:
                    st.button(session_name, 
                                            key=f"b_{session_name}_2", 
                                            on_click=link_clicked, 
                                            args=[session, global_singleton.benchmark_session_manager, False], 
                                            use_container_width=True)
            if "switch_page_bench" in st.session_state and st.session_state["switch_page_bench"]:
                st.session_state["compare_active"] = False
                st.session_state["switch_page_bench"] = False
                if global_singleton.chat_session_manager.active_session:
                    global_singleton.chat_session_manager.active_session.deinitialize()
                    global_singleton.chat_session_manager.active_session = None


                st.switch_page("pages/benchmark_eval_page.py")
            save_benchmarks = st.button("Save benchmarks", use_container_width=True, key="b_save")
            if save_benchmarks:
                global_singleton.benchmark_session_manager.save()

        else:
            st.markdown("No benchmarks")
        new_benchmark = st.button("＋ New Benchmark", use_container_width=True, key="b_create")
        if new_benchmark:
            st.session_state["compare_active"] = False
            if global_singleton.chat_session_manager.active_session:
                    global_singleton.chat_session_manager.active_session.deinitialize()
                    global_singleton.chat_session_manager.active_session = None
            if global_singleton.benchmark_session_manager.active_session:
                    global_singleton.benchmark_session_manager.active_session.deinitialize()
                    global_singleton.benchmark_session_manager.active_session = None
            st.switch_page("pages/benchmark_page.py")
        
        """ Compare benchmarks """
        if not "compare_active" in st.session_state:
            st.session_state["compare_active"] = False
        if st.session_state["compare_active"]:
            with stylable_container(
                        key="active_b_compare",
                            css_styles="""
                                button {
                                    background-color: white;
                                    color: black;
                                }
                                """,
                        ):
                st.button("Compare Benchmarks", "b_compare1", use_container_width=True)
        else:
            compare_benchmarks = st.button("Compare Benchmarks", "b_compare", use_container_width=True)
            if compare_benchmarks:
                st.session_state["compare_active"] = True
                if global_singleton.chat_session_manager.active_session:
                    global_singleton.chat_session_manager.active_session.deinitialize()
                    global_singleton.chat_session_manager.active_session = None
                if global_singleton.benchmark_session_manager.active_session:
                    global_singleton.benchmark_session_manager.active_session.deinitialize()
                    global_singleton.benchmark_session_manager.active_session = None
                st.switch_page("pages/benchmark_compare_page.py")

        """ choose llm """
        st.header("LLMs", divider="grey")
        st.markdown(f"LLM: {global_singleton.llm_model}")
        st.markdown(f"Embeddings: {global_singleton.embeddings_model}")
        st.markdown(f"Cross Encoder: {global_singleton.cross_encoder_model}")
        select_llm = st.button("Load LLMs", use_container_width=True)
        if select_llm:
            st.switch_page("pages/model_config_page.py")
