"""
Interactive dashboard for viewing Judgeval results
Shows real-time traces and evaluation metrics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os

st.set_page_config(
    page_title="Judgment Labs Agent Dashboard",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Judgment Labs Agent Evaluation Dashboard")
st.markdown("Real-time monitoring and analysis of AI agents")

# Load evaluation results
@st.cache_data
def load_results():
    if os.path.exists("evaluation_results.json"):
        with open("evaluation_results.json", "r") as f:
            return json.load(f)
    return []

results = load_results()

if not results:
    st.warning("No evaluation results found. Run `python run_evaluation.py` first.")
    st.stop()

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    
    agents = list(set(r["agent"] for r in results if "agent" in r))
    selected_agents = st.multiselect("Select Agents", agents, default=agents)
    
    date_range = st.date_input(
        "Date Range",
        value=(datetime.now() - timedelta(days=7), datetime.now()),
        max_value=datetime.now()
    )

# Main dashboard
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Scores", "⏱️ Performance", "📝 Details"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate metrics
    total_evals = len([r for r in results if r["agent"] in selected_agents])
    avg_score = sum(
        r.get("scores", {}).get("AnswerRelevancyScorer", {}).get("score", 0)
        for r in results if r["agent"] in selected_agents and "scores" in r
    ) / max(total_evals, 1)
    
    avg_time = sum(
        r.get("execution_time", 0) 
        for r in results if r["agent"] in selected_agents
    ) / max(total_evals, 1)
    
    success_rate = len([
        r for r in results 
        if r["agent"] in selected_agents and "error" not in r
    ]) / max(total_evals, 1)
    
    with col1:
        st.metric("Total Evaluations", total_evals)
    with col2:
        st.metric("Avg Relevancy Score", f"{avg_score:.2%}")
    with col3:
        st.metric("Avg Response Time", f"{avg_time:.2f}s")
    with col4:
        st.metric("Success Rate", f"{success_rate:.2%}")
    
    # Agent comparison chart
    st.subheader("Agent Performance Comparison")
    
    # Prepare data for plotting
    plot_data = []
    for result in results:
        if result["agent"] in selected_agents and "scores" in result:
            for scorer, score_data in result["scores"].items():
                if isinstance(score_data, dict):
                    plot_data.append({
                        "Agent": result["agent"],
                        "Scorer": scorer.replace("Scorer", ""),
                        "Score": score_data.get("score", 0)
                    })
    
    if plot_data:
        df_plot = pd.DataFrame(plot_data)
        fig = px.box(
            df_plot, 
            x="Agent", 
            y="Score", 
            color="Scorer",
            title="Score Distribution by Agent and Metric"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Detailed Score Analysis")
    
    # Score heatmap
    score_matrix = {}
    for result in results:
        if result["agent"] in selected_agents and "scores" in result:
            if result["agent"] not in score_matrix:
                score_matrix[result["agent"]] = {}
            
            for scorer, score_data in result["scores"].items():
                if isinstance(score_data, dict):
                    scorer_name = scorer.replace("Scorer", "")
                    if scorer_name not in score_matrix[result["agent"]]:
                        score_matrix[result["agent"]][scorer_name] = []
                    score_matrix[result["agent"]][scorer_name].append(score_data.get("score", 0))
    
    # Average scores for heatmap
    avg_matrix = []
    for agent, scores in score_matrix.items():
        row = {"Agent": agent}
        for scorer, score_list in scores.items():
            row[scorer] = sum(score_list) / len(score_list) if score_list else 0
        avg_matrix.append(row)
    
    if avg_matrix:
        df_heatmap = pd.DataFrame(avg_matrix)
        df_heatmap = df_heatmap.set_index("Agent")
        
        fig = go.Figure(data=go.Heatmap(
            z=df_heatmap.values,
            x=df_heatmap.columns,
            y=df_heatmap.index,
            colorscale="RdYlGn",
            text=df_heatmap.values.round(2),
            texttemplate="%{text}",
            textfont={"size": 12},
        ))
        
        fig.update_layout(
            title="Average Scores Heatmap",
            xaxis_title="Metrics",
            yaxis_title="Agents"
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Performance Metrics")
    
    # Response time analysis
    perf_data = []
    for result in results:
        if result["agent"] in selected_agents and "execution_time" in result:
            perf_data.append({
                "Agent": result["agent"],
                "Execution Time": result["execution_time"],
                "Question": result.get("question", "")[:50] + "..."
            })
    
    if perf_data:
        df_perf = pd.DataFrame(perf_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.violin(
                df_perf,
                y="Agent",
                x="Execution Time",
                title="Execution Time Distribution",
                orientation="h"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                df_perf,
                x="Execution Time",
                y="Agent",
                color="Agent",
                hover_data=["Question"],
                title="Execution Time by Question"
            )
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Detailed Results")
    
    # Filter and display results
    display_results = [
        r for r in results 
        if r["agent"] in selected_agents
    ]
    
    for i, result in enumerate(display_results):
        with st.expander(f"{result['agent']} - {result.get('question', 'N/A')[:50]}..."):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Question:**", result.get("question", "N/A"))
                st.write("**Execution Time:**", f"{result.get('execution_time', 0):.2f}s")
                
                if "error" in result:
                    st.error(f"Error: {result['error']}")
            
            with col2:
                if "scores" in result:
                    st.write("**Scores:**")
                    for scorer, score_data in result["scores"].items():
                        if isinstance(score_data, dict):
                            score = score_data.get("score", 0)
                            st.progress(score, text=f"{scorer}: {score:.2%}")
            
            if "answer" in result:
                st.write("**Answer Preview:**")
                st.text(result["answer"])

# Add export functionality
st.sidebar.divider()
if st.sidebar.button("📥 Export Results"):
    # Convert to CSV
    export_data = []
    for r in results:
        if "scores" in r:
            row = {
                "agent": r["agent"],
                "question": r.get("question", ""),
                "execution_time": r.get("execution_time", 0)
            }
            for scorer, score_data in r["scores"].items():
                if isinstance(score_data, dict):
                    row[scorer] = score_data.get("score", 0)
            export_data.append(row)
    
    df_export = pd.DataFrame(export_data)
    csv = df_export.to_csv(index=False)
    
    st.sidebar.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"judgeval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# Footer
st.sidebar.divider()
# Footer continuation
st.sidebar.info("""
**Judgment Labs Integration**

This dashboard is powered by Judgeval's comprehensive tracing and evaluation capabilities.

View full traces at: [platform.judgment.dev](https://platform.judgment.dev)
""")