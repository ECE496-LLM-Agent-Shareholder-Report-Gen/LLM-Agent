import streamlit as st
from streamlit_extras.stylable_container import stylable_container

""" disable active session """
def disable_active_session(global_singleton):
    st.session_state["compare_active"] = False
    st.session_state["new_chat_session"] = False
    st.session_state["new_benchmark_session"] = False
    if global_singleton.benchmark_session_manager.active_session:
            global_singleton.benchmark_session_manager.active_session.deinitialize()
            global_singleton.benchmark_session_manager.active_session = None
    if global_singleton.chat_session_manager.active_session:
            global_singleton.chat_session_manager.active_session.deinitialize()
            global_singleton.chat_session_manager.active_session = None

""" handle the user clicking a link """
def link_clicked(session, global_singleton, is_chat_session=True):
    print("setting active session to ", session.name)
    disable_active_session(global_singleton)
    if is_chat_session:
        global_singleton.chat_session_manager.active_session = session
        st.session_state["switch_page_chat"] = True
    else:
        global_singleton.benchmark_session_manager.active_session = session
        st.session_state["switch_page_bench"] = True 

""" delete session from list """
def del_session(session, global_singleton, is_chat_session=True):
    print("deleting session: ", session.name)
    if is_chat_session:
        del global_singleton.chat_session_manager.sessions[session.name]
        global_singleton.chat_session_manager.save()

    else:
        del global_singleton.benchmark_session_manager.sessions[session.name]
        global_singleton.benchmark_session_manager.save()


""" do chat sessions """
def do_chat_sessions(global_singleton):
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
                col_but, col_del = st.columns([0.8, 0.2])
                with col_but:
                    st.button(session_name, 
                                    key=f"{session_name}_2", 
                                    on_click=link_clicked, 
                                    args=[session, global_singleton], 
                                    use_container_width=True)
                with col_del:
                   st.button(":x:", key=f"{session_name}_2_del", on_click=del_session, args=[session, global_singleton, True])
      

        if "switch_page_chat" in st.session_state and st.session_state["switch_page_chat"]:
            st.session_state["switch_page_chat"] = False
            st.switch_page("pages/chat_page.py")

        save_sessions = st.button("Save sessions", use_container_width=True)
        if save_sessions:
            global_singleton.chat_session_manager.save()
    else:
        st.markdown("No sessions")
    if "new_chat_session" in st.session_state and st.session_state["new_chat_session"]:

        with stylable_container(
                        key="active_c_new_session",
                            css_styles="""
                                button {
                                    background-color: white;
                                    color: black;
                                }
                                """,
                        ):
            new_session = st.button("＋ New Chat Session", use_container_width=True, key="create1")
    else:
        new_session = st.button("＋ New Chat Session", use_container_width=True, key="create2")
    if new_session:
        disable_active_session(global_singleton)
        st.session_state["new_chat_session"] = True
        st.switch_page("pages/session_page.py")


""" do benchmarks """
def do_benchmarks(global_singleton):
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
                
                col_but, col_del = st.columns([0.8, 0.2])
                with col_but:
                    st.button(session_name, 
                                        key=f"b_{session_name}_2", 
                                        on_click=link_clicked, 
                                        args=[session, global_singleton, False], 
                                        use_container_width=True)
                with col_del:
                    st.button(":x:", key=f"b_{session_name}_2_del", on_click=del_session, args=[session, global_singleton, False])
        

        if "switch_page_bench" in st.session_state and st.session_state["switch_page_bench"]:
            st.session_state["switch_page_bench"] = False
            st.switch_page("pages/benchmark_eval_page.py")

        save_benchmarks = st.button("Save benchmarks", use_container_width=True, key="b_save")
        if save_benchmarks:
            global_singleton.benchmark_session_manager.save()

    else:
        st.markdown("No benchmarks")
    if "new_benchmark_session" in st.session_state and st.session_state["new_benchmark_session"]:

        with stylable_container(
                        key="active_b_new_session",
                            css_styles="""
                                button {
                                    background-color: white;
                                    color: black;
                                }
                                """,
                        ):
            new_benchmark = st.button("＋ New Benchmark", use_container_width=True, key="b_create1")
    else:
        new_benchmark = st.button("＋ New Benchmark", use_container_width=True, key="b_create2")

            
    if new_benchmark:
        disable_active_session(global_singleton)
        st.session_state["new_benchmark_session"] = True
        st.switch_page("pages/benchmark_page.py")


""" do benchmark compare """
def do_benchmark_compare(global_singleton):
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
            disable_active_session(global_singleton)
            st.session_state["compare_active"] = True
           
            st.switch_page("pages/benchmark_compare_page.py")

""" choose llm """
def do_llm_select(global_singleton):
    st.header("LLMs", divider="grey")
    st.markdown(f"LLM: {global_singleton.llm_model}")
    st.markdown(f"Embeddings: {global_singleton.embeddings_model}")
    st.markdown(f"Cross Encoder: {global_singleton.cross_encoder_model}")
    select_llm = st.button("Load LLMs", use_container_width=True)
    if select_llm:
        st.switch_page("pages/model_config_page.py")


""" navbar """
def navbar(global_singleton):
    # Sidebar
    with st.sidebar:
        #st.rerun()
        #print("girid")
        do_chat_sessions(global_singleton)
        do_benchmarks(global_singleton)
        do_benchmark_compare(global_singleton)
        do_llm_select(global_singleton)
        

       