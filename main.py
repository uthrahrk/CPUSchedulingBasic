import streamlit as st
import random
import numpy as np
import pandas as pd
import plotly.graph_objs as go

class Process:
    def _init_(self, name, arrival_time, burst_time, priority=1):
        self.name = name
        self.arrival_time = arrival_time
        self.burst_time = burst_time
        self.priority = priority
        self.remaining_time = burst_time
        self.completion_time = 0
        self.waiting_time = 0
        self.turnaround_time = 0
        self.response_time = -1
        self.start_time = None

    def calculate_metrics(self, current_time):
        if self.start_time is None:
            self.start_time = current_time
        self.completion_time = current_time
        self.turnaround_time = self.completion_time - self.arrival_time
        self.waiting_time = max(0, self.turnaround_time - self.burst_time)
        self.response_time = max(0, self.start_time - self.arrival_time)

class SchedulingAlgorithm:
    @staticmethod
    def fcfs(processes):
        processes = [Process(p.name, p.arrival_time, p.burst_time, p.priority) for p in processes]
        processes.sort(key=lambda p: p.arrival_time)
        
        current_time = 0
        gantt_chart = []

        for process in processes:
            current_time = max(current_time, process.arrival_time)
            process.start_time = current_time
            process.calculate_metrics(current_time + process.burst_time)
            gantt_chart.append((process.name, current_time, current_time + process.burst_time))
            current_time += process.burst_time

        return processes, gantt_chart

    @staticmethod
    def sjf(processes, preemptive=False):
        processes = [Process(p.name, p.arrival_time, p.burst_time, p.priority) for p in processes]
        processes.sort(key=lambda p: p.arrival_time)
        
        current_time = 0
        gantt_chart = []
        completed = []

        while processes:
            available_processes = [p for p in processes if p.arrival_time <= current_time]
            
            if not available_processes:
                current_time += 1
                continue

            if preemptive:
                selected_process = min(available_processes, key=lambda p: p.remaining_time)
                
                if selected_process.start_time is None:
                    selected_process.start_time = current_time
                
                gantt_chart.append((selected_process.name, current_time, current_time + 1))
                selected_process.remaining_time -= 1
                current_time += 1

                if selected_process.remaining_time == 0:
                    selected_process.calculate_metrics(current_time)
                    completed.append(selected_process)
                    processes.remove(selected_process)
            else:
                selected_process = min(available_processes, key=lambda p: p.burst_time)
                
                current_time = max(current_time, selected_process.arrival_time)
                selected_process.start_time = current_time
                
                selected_process.calculate_metrics(current_time + selected_process.burst_time)
                gantt_chart.append((selected_process.name, current_time, current_time + selected_process.burst_time))
                
                current_time += selected_process.burst_time
                completed.append(selected_process)
                processes.remove(selected_process)

        return completed, gantt_chart

    @staticmethod
    def round_robin(processes, time_quantum):
        processes = [Process(p.name, p.arrival_time, p.burst_time, p.priority) for p in processes]
        processes.sort(key=lambda p: p.arrival_time)
        
        current_time = 0
        gantt_chart = []
        queue = processes.copy()
        completed = []

        while queue:
            process = queue.pop(0)
            
            if process.start_time is None:
                process.start_time = current_time

            execute_time = min(time_quantum, process.remaining_time)
            gantt_chart.append((process.name, current_time, current_time + execute_time))
            
            current_time += execute_time
            process.remaining_time -= execute_time

            if process.remaining_time == 0:
                process.calculate_metrics(current_time)
                completed.append(process)
            else:
                queue.append(process)

        return completed, gantt_chart

    @staticmethod
    def calculate_metrics(processes):
        if not processes:
            return {}
        
        total_completion_time = max(p.completion_time for p in processes)
        total_burst_time = sum(p.burst_time for p in processes)
        
        return {
            'Avg Waiting Time': round(np.mean([p.waiting_time for p in processes]), 2),
            'Avg Turnaround Time': round(np.mean([p.turnaround_time for p in processes]), 2),
            'Avg Response Time': round(np.mean([p.response_time for p in processes]), 2),
            'Total Completion Time': total_completion_time,
            'CPU Utilization': round((total_burst_time / total_completion_time) * 100, 2) if total_completion_time > 0 else 0,
            'Throughput': round(len(processes) / total_completion_time, 2) if total_completion_time > 0 else 0
        }

def create_comparative_charts(results):
    # Prepare data for charts
    algo_names = list(results.keys())
    metrics = [
        'Avg Waiting Time', 
        'Avg Turnaround Time', 
        'Avg Response Time', 
        'CPU Utilization', 
        'Throughput'
    ]

    # Bar Chart for Performance Metrics
    chart_data = {metric: [results[algo]['metrics'][metric] for algo in algo_names] for metric in metrics}
    
    fig1 = go.Figure()
    for metric in metrics:
        fig1.add_trace(go.Bar(
            x=algo_names, 
            y=chart_data[metric], 
            name=metric
        ))
    
    fig1.update_layout(
        title='Comparative Performance Metrics',
        xaxis_title='Scheduling Algorithm',
        yaxis_title='Metric Value',
        barmode='group'
    )

    # Gantt Chart Comparison
    fig2 = go.Figure()
    for algo_name, result in results.items():
        for process_name, start, end in result['gantt_chart']:
            fig2.add_trace(go.Bar(
                x=[algo_name],
                y=[end - start],
                base=start,
                name=f'{process_name} ({algo_name})',
                orientation='h'
            ))
    
    fig2.update_layout(
        title='Process Execution Timeline Comparison',
        xaxis_title='Scheduling Algorithm',
        yaxis_title='Time',
        barmode='stack'
    )

    return fig1, fig2

def main():
    st.set_page_config(page_title="CPU Scheduling Simulator", layout="wide")
    st.title("🖥 CPU Scheduling Simulator")

    # Sidebar Configuration
    st.sidebar.header("Simulation Parameters")
    
    # Input Method Selection
    input_method = st.sidebar.radio(
        "Input Method", 
        ["Random Generation", "Manual Input"], 
        index=0
    )
    
    # Number of Processes
    num_processes = st.sidebar.slider(
        "Number of Processes", 
        min_value=1, 
        max_value=10, 
        value=4
    )
    
    # Time Quantum for Round Robin
    time_quantum = st.sidebar.slider(
        "Time Quantum (for Round Robin)", 
        min_value=1, 
        max_value=10, 
        value=4
    )

    # Process Generation
    processes = []
    if input_method == "Random Generation":
        processes = [
            Process(
                f"P{i+1}", 
                random.randint(0, 5),   # Arrival Time
                random.randint(1, 10),  # Burst Time
                random.randint(1, 5)    # Priority
            ) for i in range(num_processes)
        ]
    else:
        # Manual Input
        st.sidebar.subheader("Process Details")
        for i in range(num_processes):
            col1, col2, col3 = st.sidebar.columns(3)
            with col1:
                arrival_time = st.number_input(
                    f"P{i+1} Arrival Time", 
                    min_value=0, 
                    value=0, 
                    key=f"arrival_{i}_unique"
                )
            with col2:
                burst_time = st.number_input(
                    f"P{i+1} Burst Time", 
                    min_value=1, 
                    value=1, 
                    key=f"burst_{i}_unique"
                )
            with col3:
                priority = st.number_input(
                    f"P{i+1} Priority", 
                    min_value=1, 
                    value=1, 
                    key=f"priority_{i}_unique"
                )
            
            processes.append(Process(f"P{i+1}", arrival_time, burst_time, priority))

    # Scheduling Algorithms
    algorithms = {
        "FCFS": SchedulingAlgorithm.fcfs,
        "SJF (Non-Preemptive)": lambda p: SchedulingAlgorithm.sjf(p, preemptive=False),
        "SJF (Preemptive)": lambda p: SchedulingAlgorithm.sjf(p, preemptive=True),
        "Round Robin": lambda p: SchedulingAlgorithm.round_robin(p, time_quantum),
    }

    # Algorithm Selection
    selected_algorithms = st.multiselect(
        "Select Scheduling Algorithms", 
        list(algorithms.keys()), 
        default=["FCFS"]
    )

    # Run Simulation Button
    if st.button("Run Simulation", type="primary"):
        # Validate Inputs
        if not processes:
            st.error("No processes defined. Please add processes.")
            return

        if not selected_algorithms:
            st.error("Please select at least one scheduling algorithm.")
            return

        # Create Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Process Details", 
            "📈 Performance Metrics", 
            "🔍 Comparative Analysis",
            "📉 Comparative Charts"
        ])

        # Results Storage
        results = {}

        with tab1:
            # Process Details and Gantt Charts
            for algo_name in selected_algorithms:
                st.subheader(f"Algorithm: {algo_name}")
                
                try:
                    # Run Algorithm
                    completed_processes, gantt_chart = algorithms[algo_name](processes.copy())
                    
                    # Calculate Metrics
                    metrics = SchedulingAlgorithm.calculate_metrics(completed_processes)
                    
                    # Store Results
                    results[algo_name] = {
                        'processes': completed_processes,
                        'metrics': metrics,
                        'gantt_chart': gantt_chart
                    }

                    # Display Process Details
                    process_df = pd.DataFrame([{
                        'Name': p.name,
                        'Arrival Time': p.arrival_time,
                        'Burst Time': p.burst_time,
                        'Completion Time': p.completion_time,
                        'Turnaround Time': p.turnaround_time,
                        'Waiting Time': p.waiting_time,
                        'Response Time': p.response_time
                    } for p in completed_processes])
                    st.dataframe(process_df, use_container_width=True)

                except Exception as e:
                    st.error(f"Error in {algo_name} algorithm: {e}")

        with tab2:
            # Performance Metrics
            for algo_name, result in results.items():
                st.subheader(f"{algo_name} Metrics")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Avg Waiting Time", result['metrics']['Avg Waiting Time'])
                    st.metric("Avg Turnaround Time", result['metrics']['Avg Turnaround Time'])
                    st.metric("CPU Utilization", f"{result['metrics']['CPU Utilization']}%")
                
                with col2:
                    st.metric("Avg Response Time", result['metrics']['Avg Response Time'])
                    st.metric("Total Completion Time", result['metrics']['Total Completion Time'])
                    st.metric("Throughput", result['metrics']['Throughput'])

        with tab3:
            # Comparative Analysis Table
            if len(results) > 1:
                comparison_df = pd.DataFrame.from_dict(
                    {algo: result['metrics'] for algo, result in results.items()}, 
                    orient='index'
                )
                st.dataframe(comparison_df, use_container_width=True)
            else:
                st.warning("Select multiple algorithms for comparative analysis")

        with tab4:
            # Comparative Charts
            if len(results) > 1:
                # Create and display comparative charts
                performance_chart, timeline_chart = create_comparative_charts(results)
                
                st.plotly_chart(performance_chart, use_container_width=True)
                st.plotly_chart(timeline_chart, use_container_width=True)
            else:
                st.warning("Select multiple algorithms for comparative charts")

if _name_ == "_main_":
    main()
