from EMDMeasurment.ComparisonMethods import produce_visualizations_from_event_logs_paths, show_all_visualizations


if __name__=="__main__":
    ret_dict = produce_visualizations_from_event_logs_paths("C:/running-example.xes", "C:/running-example.xes")
    show_all_visualizations(ret_dict)
