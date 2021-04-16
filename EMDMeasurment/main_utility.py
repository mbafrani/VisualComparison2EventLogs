from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from EMDMeasurment.SMS import SMS
from pm4py.objects.log.importer.xes import importer as xes_importer_factory
import time
from EMDMeasurment.EMD import EMD
import pandas as pd


class CompareConf:

    def compare_conf(self,event_log,sim_log):
        sensitive = []
        time_accuracy = "hours"  # original, seconds, minutes, hours, days
        event_attributes = ['concept:name']
        # these life cycles are applied only when all_lif_cycle = False
        life_cycle = ['complete', '', 'COMPLETE']
        # when life cycle is in trace attributes then all_life_cycle has to be True
        all_life_cycle = False  # True will ignore the transitions specified in life_cycle

        original_log = event_log
        privacy_log = sim_log

        sms = SMS()
        logsimple, traces, sensitives = sms.create_simple_log_adv(original_log, event_attributes, life_cycle,
                                                                  all_life_cycle, sensitive, time_accuracy)
        logsimple_2, traces_2, sensitives_2 = sms.create_simple_log_adv(privacy_log, event_attributes, life_cycle,
                                                                        all_life_cycle, sensitive, time_accuracy)

        activities1 = sms.get_unique_act(traces)
        activities2 = sms.get_unique_act(traces_2)
        uniq_activities = activities2.union(activities1)
        map_dict_act_chr, map_dict_chr_act, uniq_char = sms.map_act_char(uniq_activities)

        # log 1 convert to char
        simple_log_char_1 = sms.convert_simple_log_act_to_char(traces, map_dict_act_chr)

        # log 2 convert to char
        simple_log_char_2 = sms.convert_simple_log_act_to_char(traces_2, map_dict_act_chr)

        start_time = time.time()

        my_emd = EMD()
        # log_freq_1, log_only_freq_1 = my_emd.log_freq(traces)
        # log_freq_2 , log_only_freq_2 = my_emd.log_freq(traces_2)

        log_freq_1, log_only_freq_1 = my_emd.log_freq(simple_log_char_1)
        log_freq_2, log_only_freq_2 = my_emd.log_freq(simple_log_char_2)

        cost_lp,df_distance = my_emd.emd_distance_pyemd(log_only_freq_1, log_only_freq_2, log_freq_1, log_freq_2)
        # cost_lp = my_emd.emd_distance(log_freq_1,log_freq_2)

        data_utility = 1 - cost_lp
        print("data_utility# %0.3f" % (data_utility))
        return data_utility, traces, traces_2,df_distance
