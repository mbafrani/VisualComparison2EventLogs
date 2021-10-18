from collections import Counter
import pm4py
import pandas as pd
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from EMDMeasurment.main_utility import CompareConf
import numpy as np
from pm4py.statistics.performance_spectrum import algorithm as performance_spectrum
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from pm4py.statistics.traces.log import case_statistics
from pm4py.objects.log.importer.xes import importer as xes_importer
from collections import defaultdict
from tempfile import NamedTemporaryFile
from pm4py.util import vis_utils
import os
import uuid


class Compare():

    def convert_log(self, el_address):
        log_csv = pd.read_csv(el_address)
        log_csv = dataframe_utils.convert_timestamp_columns_in_df(log_csv)
        log_csv = log_csv.sort_values('time:timestamp')
        event_log = log_converter.apply(log_csv)
        return event_log

    def create_dfg(self, event_log, sim_log):
        activities_count = pm4py.get_event_attribute_values(event_log, "concept:name")
        act_list = list(activities_count.keys())
        event_log_dfg = pd.DataFrame(0, columns=act_list, index=act_list)
        event_log_dfg_count = dfg_discovery.apply(event_log)
        for act, act_freq in event_log_dfg_count.items():
            event_log_dfg[act[1]][act[0]] = act_freq

        activities_count = pm4py.get_event_attribute_values(sim_log, "concept:name")
        act_list = list(activities_count.keys())
        event_log_sim_dfg = pd.DataFrame(0, columns=act_list, index=act_list)
        event_log_sim_dfg_count = dfg_discovery.apply(sim_log)
        for act, act_freq in event_log_sim_dfg_count.items():
            event_log_sim_dfg[act[1]][act[0]] = act_freq

        diff_two_dfg = event_log_dfg - event_log_sim_dfg
        return event_log_dfg, event_log_sim_dfg, diff_two_dfg

    def performance(self, event_log, sim_log, tw):
        all_case_durations_el = case_statistics.get_all_casedurations(event_log)
        real_avg_serv_time = np.mean(all_case_durations_el) / 3600
        all_case_durations_sim = case_statistics.get_all_casedurations(sim_log)
        sim_avg_serv_time = np.mean(all_case_durations_sim) / 3600
        Measures = ['First Event Log', 'Second Event Log']
        values = [real_avg_serv_time, sim_avg_serv_time]
        conf_plt = plt.bar(Measures, values)
        plt.xlabel('Average Service Time of Cases')
        plt.ylabel("Hours")
        return real_avg_serv_time, sim_avg_serv_time

    def spectrum(self, event_log, sim_log, dfgs):
        # dfgs= self.create_dfg(event_log,sim_log)
        event_log_dfg = dfgs[0]
        max_event_log_freq = np.max(np.max(event_log_dfg))
        sim_log_dfg = dfgs[1]
        max_sim_log_freq = np.max(np.max(sim_log_dfg))
        for i in event_log_dfg.columns:
            for j in event_log_dfg.columns:
                if event_log_dfg[j][i] / max_event_log_freq > 0.05:
                    event_log_dfg[j][i] = np.mean(performance_spectrum.apply(event_log, [i, j],
                                                                             parameters={
                                                                                 performance_spectrum.Parameters.ACTIVITY_KEY: "concept:name",
                                                                                 performance_spectrum.Parameters.TIMESTAMP_KEY: "time:timestamp"})[
                                                      'points']) / 3600
                else:
                    event_log_dfg[j][i] = 0
                if sim_log_dfg[j][i] / max_sim_log_freq > 0.05:
                    sim_log_dfg[j][i] = np.mean(performance_spectrum.apply(sim_log, [i, j],
                                                                           parameters={
                                                                               performance_spectrum.Parameters.ACTIVITY_KEY: "concept:name",
                                                                               performance_spectrum.Parameters.TIMESTAMP_KEY: "time:timestamp"})[
                                                    'points']) / 3600
                else:
                    sim_log_dfg[j][i] = 0
        diff_dfg_perf = event_log_dfg - sim_log_dfg
        diff_dfg_perf = diff_dfg_perf.fillna(0)
        # sns.heatmap(diff_dfg_perf, annot=True)

        return event_log_dfg, sim_log_dfg, diff_dfg_perf

    def conformance(self, event_log, sim_log, conformance_file1, conformance_file2):
        compareconf = CompareConf()
        emd_measure, case_real, case_sim, df_distance = compareconf.compare_conf(event_log, sim_log)
        trace_real = list(set(case_real))
        trace_sim = list(set(case_sim))
        trace_diff_list = list(set(trace_real) - set(trace_sim))
        trace_diff_list_sim = list(set(trace_sim) - set(trace_real))
        traces_similar_list = list(set(trace_real).intersection(trace_sim))
        percentage_trace_diff = (len(trace_diff_list) + len(trace_diff_list_sim)) / (
                len(trace_diff_list) + len(traces_similar_list) + len(trace_diff_list_sim))
        # len(trace_diff_list)/(len(trace_diff_list)+len(traces_similar_list))
        new_behavior_miss_from_origin = Counter(trace_sim) - Counter(trace_real)
        removed_behavior_miss_from_sim = Counter(trace_real) - Counter(trace_sim)
        important_real = Counter(case_real).most_common()
        sorted_real_trace = [j[0] for j in important_real]
        important_sim = Counter(case_sim).most_common()
        sorted_sim_trace = [j[0] for j in important_sim]
        percentage_new_beh = len(new_behavior_miss_from_origin) / len(trace_sim)
        percentage_removed_beh = len(removed_behavior_miss_from_sim) / len(trace_real)
        conf_var_related_metr_list = [len(traces_similar_list), len(new_behavior_miss_from_origin),
                                      len(removed_behavior_miss_from_sim)]

        self.visulaize_EMD_detial(df_distance, sorted_real_trace, sorted_sim_trace, conformance_file1)
        self.visualize_pairwise_variants(case_real, case_sim, conformance_file2)

        return round(1 - emd_measure,
                     2), percentage_new_beh, percentage_removed_beh, percentage_trace_diff, conf_var_related_metr_list, case_real, case_sim

    def visulaize_EMD_detial(self, df_distance, trace_real, trace_sim, conformance_file1):
        df_distance = df_distance[0:len(trace_real)]
        df_distance = df_distance[df_distance.columns[0:len(trace_sim)]]
        df_distance = df_distance[(df_distance.T != 0).any()]
        df_distance = df_distance.apply(lambda x: x / x.sum(), axis=1)
        # tr = [list(i).insert(5, '<br>') for i in trace_sim if len(i) > 5]
        if len(df_distance.index) > 30:
            df_distance = df_distance[0:20]
        fig = go.Figure()
        py_color = ['red', 'blue', 'green', 'darkblue', 'yellow', 'purple', 'pink', 'brown',
                    'orange', 'gold', 'red', 'blue', 'green', 'darkblue', 'yellow', 'purple', 'pink', 'brown', 'orange',
                    'gold',
                    'purple', 'pink', 'brown', 'orange', 'gold', 'red', 'blue', 'green', 'darkblue', 'yellow', 'purple',
                    'pink', 'brown',
                    'orange', 'gold', 'red', 'blue', 'green', 'darkblue', 'yellow', 'purple', 'pink', 'brown', 'orange',
                    'gold',
                    'purple', 'pink', 'brown', 'orange', 'gold']
        col_list = []
        for i in range(len(df_distance.index)):
            j = 0
            while j < len(df_distance.columns):
                col_list.append(i)
                j = j + 1

        for i in df_distance.index:
            name = 'var' + str(i)  # + str(trace_real[i])
            y, x = np.meshgrid(i, df_distance.columns)
            size1 = df_distance.iloc[i].values.flatten() * 100
            size = np.array(size1)
            size = [round(i, 2) for i in size]
            hovertxt = []
            trc_color = []
            for j in range(0, len(trace_sim)):
                diff1 = list((Counter(trace_real[i]) - Counter(trace_sim[j])).elements())
                diff2 = list((Counter(trace_sim[j]) - Counter(trace_real[i])).elements())
                # if len(set(trace_real[i]).symmetric_difference(set(trace_sim[j]))) != 0:
                if len(diff1) == 0 and len(diff2) == 0:
                    for k in range(0, min(len(trace_real[i]), len(trace_sim[j]))):
                        if trace_real[i][k] != trace_sim[j][k]:
                            diff1.append(trace_real[i][k])
                            diff1.append(trace_sim[j][k])
                if len(diff1) != 0 or len(diff2) != 0:
                    # trace_real[i]).symmetric_difference(set(trace_sim[j])
                    txt = str(size[j]) + '<br><b> Diff:' + str(set(diff1)) + str(set(diff2)) + '</b><br>' + str(
                        trace_sim[j]) + '<br>' + str(trace_real[i])
                    trc_color.append(py_color[i])

                elif len(diff1) == 0 and len(diff2) == 0:
                    txt = str(size[j]) + '<br>Similar Trace' + '<br>' + str(trace_real[i])
                    col_list[i * len(trace_sim) + j] = -1
                    # trc_color=[i]
                    # trc_color=trc_color*len(trace_sim)
                    trc_color.append('black')

                tr_sim_color = trc_color
                hovertxt.append(txt)
            # col_list[i * len(trace_sim):i * len(trace_sim)+len(trace_sim)]
            # hovertxt= [str(size[j])+str(trace_sim[j])+str('\n')+str(set(trace_real[i])-set(trace_sim[j]))for j in range(0,len(trace_sim))]
            fig.add_trace(go.Scatter(
                x=x.flatten(), y=y.flatten(),
                text=size,
                mode='markers+lines',
                name=name,
                hovertext=hovertxt,
                hoverinfo='text',
                # opacity=0.2,
                marker_size=np.divide(size, 3 / 2),
                # marker_size=size,
                line=dict(color=py_color[i], dash='dash'),
                marker=dict(size=size, color=tr_sim_color, opacity=0.7)))

        fig.update_layout(
            title="Earth Mover Distance",
            margin=dict(l=0, r=0, b=0, t=30),
            yaxis_title="Frequent Variants (Real Event Log)",
            xaxis_title="Frequent Variants (Simulated Event Log)",
            legend_title="Real Event Log Variants:",
            font=dict(size=12, color="RebeccaPurple"),
            legend=dict(orientation="v",
                        font=dict(
                            family="Courier",
                            size=10)

                        )
        )

        # fig.show()
        fig.write_html(conformance_file1)
        return

    def visualize_pairwise_variants(self, case_real, case_sim, conformance_file2):
        fig = go.Figure()
        important_real = Counter(case_real).most_common()
        percentages_rea = {x: round(float(y) / len(case_real) * 100, 2) for x, y in important_real}
        important_sim = Counter(case_sim).most_common()
        percentages_sim = {x: round((float(y) / len(case_sim) * 100), 2) for x, y in important_sim}
        rek = set(list(percentages_rea)[0:10])
        res = set(list(percentages_sim)[0:10])

        percentages_rea = percentages_rea
        rek = set(list(percentages_rea)[0:10])
        res = set(list(percentages_sim)[0:10])
        h = defaultdict(list)

        for k in list(percentages_rea):
            h[k].append(percentages_rea.get(k))
            if k in list(percentages_sim):
                h[k].append(percentages_sim.get(k))
            else:
                h[k].append(0)
        for k in list(percentages_sim):
            if k not in list(percentages_rea):
                h[k].append(0)
                h[k].append(percentages_sim.get(k))

        fig = go.Figure()
        fig.add_trace(
            go.Funnel(
                name='Real Event Log',
                orientation="h",
                hovertext=list(h.keys()),
                hoverinfo='text',
                # y=list(range(0, len(xlis))),
                y=list(range(0, len(list(h.keys())))),
                x=[i[0] for i in h.values()],
                textposition="inside", marker={"color": "blue"}))
        fig.add_trace(go.Funnel(
            name='Simulated',
            orientation="h",
            hovertext=list(h.keys()),
            hoverinfo='text',
            # y=list(range(0, len(xlis))),
            y=list(range(0, len(list(h.keys())))),
            x=[i[1] for i in h.values()],
            textposition="inside", marker={"color": "purple"}))

        fig.update_layout(title="Compare Frequency of Variants",
                          yaxis_title="Most Frequent Variants in the Real Event Log",
                          xaxis_title="Frequency of Variants (%)",
                          # annotations=[go.layout.Annotation(text=str(tx), align='right',   showarrow=False, xref='paper',yref='paper',x=1,y=0)],
                          legend_title="Real Event Log Variants",
                          font=dict(size=20, color="RebeccaPurple"), legend=dict(orientation="h"))
        # fig.show()
        fig.write_html(conformance_file2)

        return

    def performance_change_graph(self, event_log_dfg_list):

        plt.figure()
        x, y = np.meshgrid(event_log_dfg_list[0].columns, event_log_dfg_list[0].index)
        size = event_log_dfg_list[0].values.flatten() * 0.01
        fig = go.Figure(data=[go.Scatter(
            x=x.flatten(),
            y=y.flatten(),
            mode='markers',
            marker=dict(
                size=size,
                sizemode='area',
                sizeref=2. * max(size) / (40. ** 2),
                sizemin=4
            )
        )])

        fig.show()

        x, y = np.meshgrid(event_log_dfg_list[0].columns, event_log_dfg_list[0].index)
        event_log_dfg_list[0] *= 0.1
        plt.scatter(x=x.flatten(), y=y.flatten(), s=event_log_dfg_list[0].values.flatten(), c='blue', edgecolors='blue',
                    alpha=0.05)

        x, y = np.meshgrid(event_log_dfg_list[1].columns, event_log_dfg_list[1].index)
        event_log_dfg_list[1] *= 0.1
        plt.scatter(x=x.flatten(), y=y.flatten(), s=event_log_dfg_list[1].values.flatten(), c='red', edgecolors='red',
                    alpha=0.065)

        # plt.show()
        diff_dfgs_perf = (event_log_dfg_list[0] - event_log_dfg_list[1]) / 3600
        diff_dfgs_perf = diff_dfgs_perf.fillna(0)

        sns.heatmap(diff_dfgs_perf, annot=True, cmap="YlGnBu")
        plt.suptitle('Performance Changes')

        return

    def conf_plot(self, conf_metrics, conformance_plt3, conformance_plt4):
        # Measures = ['EMD Similarity', 'New Behavior', 'Removed Bahavior', 'Change in #Variants']
        Measures = ['EMD Similarity', 'New Behavior', 'Removed Bahavior']
        # values = [conf_metrics[0], conf_metrics[1], conf_metrics[2], (conf_metrics[3])]
        values = [conf_metrics[0], conf_metrics[1], conf_metrics[2]]
        conf_plt = plt.bar(Measures, values)
        plt.xlabel('Measures')
        plt.ylabel("Percentage")
        # plt.show()
        plt.savefig(conformance_plt3)

        conf_metrics[4]
        labels = 'Similar Behavior', 'New Behavior', 'Removed Bahavior'
        labels_exp = ['Intersection of Two Event Logs', 'Variants only in the Simulated Log',
                      'Variants only in the Real Event Log']
        sizes = [conf_metrics[4][0], conf_metrics[4][1], conf_metrics[4][2]]
        explode = (0, 0.1, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, colors=['skyblue', 'purple', 'pink'], autopct='%1.1f%%',
                shadow=True, startangle=90)
        plt.legend(labels_exp, loc="best")
        plt.title('Comparing Two Logs (based on variants)')
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        # plt.show()
        plt.savefig(conformance_plt4)

        return

    def spectrum_visualize(self, event_log, sim_log, event_log_dfg_list, spectrum_file):
        dfgs = self.create_dfg(event_log, sim_log)
        event_log_dfg_list[0].sort_index(axis=0, inplace=True)
        event_log_dfg_list[0].sort_index(axis=1, inplace=True)
        event_log_dfg_list[1].sort_index(axis=0, inplace=True)
        event_log_dfg_list[1].sort_index(axis=1, inplace=True)
        x, y = np.meshgrid(event_log_dfg_list[0].columns, event_log_dfg_list[0].index)
        size = event_log_dfg_list[0].values.flatten() / 3600
        sizeorg = [round(i, 4) for i in size]
        event_log_freq = dfgs[0]
        event_log_freq = event_log_freq.values.flatten()
        opac = abs((event_log_freq - np.min(event_log_freq)) / (np.max(event_log_freq) - np.min(event_log_freq)))
        txtsize = [str(i) for i in sizeorg]
        txtofreq = [str(round(j, 4)) for j in opac]
        txt = [txtsize[i] + ", Freq:" + txtofreq[i] for i in range(0, len(txtsize))]
        fig1 = go.Scatter(x=y.flatten(), y=x.flatten(), mode='markers', text=txt,
                          marker=dict(color='blue', size=sizeorg, sizemode='area',
                                      sizeref=2. * max(sizeorg) / (50. ** 2),
                                      sizemin=4,
                                      opacity=opac,
                                      ), name='Real Process'
                          )
        x, y = np.meshgrid(event_log_dfg_list[1].columns, event_log_dfg_list[1].index)
        size = event_log_dfg_list[1].values.flatten() / 3600
        event_log_freq = dfgs[1]
        event_log_freq = event_log_freq.values.flatten()
        opacsim = abs((event_log_freq - np.min(event_log_freq)) / (np.max(event_log_freq) - np.min(event_log_freq)))
        txtsize = [str(i) for i in size]
        txtofreq = [str(round(j, 2)) for j in opacsim]
        txt = [txtsize[i] + ", Freq:" + txtofreq[i] for i in range(0, len(txtsize))]
        fig2 = go.Scatter(
            x=y.flatten(),
            y=x.flatten(),
            mode='markers',
            text=txt,
            opacity=0.5,
            marker=dict(
                color='yellow',
                size=size,
                sizemode='area',
                sizeref=2. * max(size) / (50. ** 2),
                sizemin=4,
                opacity=opacsim
            )
            , name='Simulated Process'
        )
        diff = (event_log_dfg_list[0] - event_log_dfg_list[1])
        x, y = np.meshgrid(diff.columns, diff.index)
        size = abs(diff.values.flatten()) / 3600
        event_log_freq = dfgs[2]
        event_log_freq = event_log_freq.values.flatten()
        opacdif = abs((event_log_freq - np.min(event_log_freq)) / (np.max(event_log_freq) - np.min(event_log_freq)))
        opacdif = np.where(np.isnan(opacdif), 0, opacdif)
        txtsize = [str(i) for i in size]
        txtofreq = [str(round(j, 2)) for j in opacdif]
        txt = [txtsize[i] + ", Freq:" + txtofreq[i] for i in range(0, len(txtsize))]
        fig3 = go.Scatter(
            x=y.flatten(),
            y=x.flatten(),
            mode='markers',
            text=txt,
            opacity=0.3,
            marker=dict(
                color='red',
                size=size,
                sizemode='area',
                # sizeref=2. * max(size) / (50. ** 2),
                sizemin=4,
                opacity=opacdif
            ), name='Difference'
        )
        data = [fig1, fig2, fig3]
        fig = go.Figure(data=data)
        # fig.show()
        fig.write_html(spectrum_file)
        return

    def variant_importance_vis(self, case_real, case_sim):
        trace_real_count = Counter(case_real)
        trace_sim_count = Counter(case_sim)
        trace_real = list(set(case_real))
        trace_sim = list(set(case_sim))
        traces_similar_list = list(set(trace_real).intersection(trace_sim))
        # trace_real_count = [(i, trace_real_count[i] / len(case_real) * 100.0) for i in trace_real_count]
        # trace_sim_count = [(i, trace_sim_count[i] / len(case_sim) * 100.0) for i in trace_sim_count]

        fig, ax = plt.subplots()
        for i in traces_similar_list:
            ax.plot([1, 2],
                    [100 * trace_real_count.get(i) / len(case_real), 100 * trace_sim_count.get(i) / len(case_sim)],
                    marker="o", label=str(i))
            plt.fill_between([1, 2], [100 * trace_real_count.get(i) / len(case_real),
                                      100 * trace_sim_count.get(i) / len(case_sim)],
                             min(trace_real_count.get(i) / len(case_real), trace_sim_count.get(i) / len(case_sim)),
                             alpha=0.30)

        plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=2, prop={'size': 8})
        ax2 = ax.twinx()
        ax.set_ylabel("Real Event Log %", color="blue", fontsize=14)
        ax2.set_ylabel("Simulated Event Log %", color="purple", fontsize=14)
        ax.set_ylim(0, 10 + 100 * (Counter(case_real).most_common()[0][1]) / len(case_real))
        ax2.set_ylim(0, 10 + 100 * (Counter(case_sim).most_common()[0][1]) / len(case_sim))
        ax.set_xlim()
        plt.show()
        return

    def preprocess_logs(self, event_log, sim_log):
        if event_log.split('.')[-1] == 'csv':
            event_log = self.convert_log(event_log)
        elif event_log.split('.')[-1] == 'xes':
            event_log = xes_importer.apply(event_log)
        if sim_log.split('.')[-1] == 'csv':
            sim_log = self.convert_log(sim_log)
        elif sim_log.split('.')[-1] == 'xes':
            sim_log = xes_importer.apply(sim_log)

        return event_log, sim_log


def get_full_path(file_name):
    return os.path.join("static", "temp", os.path.basename(os.path.normpath(file_name)))


def get_temp_file_name(extension):
    return get_full_path(str(uuid.uuid4())+extension)


def produce_visualizations_from_event_logs_paths(path1, path2):
    conformance_file1 = get_temp_file_name(".html")
    conformance_file2 = get_temp_file_name(".html")

    conformance_plt3 = get_temp_file_name(".svg")
    conformance_plt4 = get_temp_file_name(".svg")

    compare = Compare()
    event_log, sim_log = compare.preprocess_logs(path1, path2)
    conf_metrics = compare.conformance(event_log, sim_log, conformance_file1, conformance_file2)
    compare.conf_plot(conf_metrics, conformance_plt3, conformance_plt4)
    # real_avg_serv_time, sim_avg_serv_time=compare.performance(event_log,sim_log,'1D')
    dfgs = compare.create_dfg(event_log, sim_log)
    perf_dfgs = compare.spectrum(event_log, sim_log, dfgs)

    spectrum_file = get_temp_file_name(".html")

    compare.spectrum_visualize(event_log, sim_log, perf_dfgs, spectrum_file)

    ret_dict = {}
    ret_dict["conformance_file1"] = conformance_file1
    ret_dict["conformance_file2"] = conformance_file2
    ret_dict["conformance_plt3"] = conformance_plt3
    ret_dict["conformance_plt4"] = conformance_plt4
    ret_dict["spectrum_file"] = spectrum_file

    return ret_dict


def show_all_visualizations(ret_dict):
    vis_utils.open_opsystem_image_viewer(get_full_path(ret_dict["conformance_file1"]))
    vis_utils.open_opsystem_image_viewer(get_full_path(ret_dict["conformance_file2"]))
    vis_utils.open_opsystem_image_viewer(get_full_path(ret_dict["conformance_plt3"]))
    vis_utils.open_opsystem_image_viewer(get_full_path(ret_dict["conformance_plt4"]))
    vis_utils.open_opsystem_image_viewer(get_full_path(ret_dict["spectrum_file"]))
