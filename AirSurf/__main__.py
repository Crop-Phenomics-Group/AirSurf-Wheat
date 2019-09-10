from pipeline.pipeline import Pipeline
# from interface.interface import Interface

# Pipeline.getInstance().run_pipeline(".", img_path="/Users/bauera/work/airsurf/wheat/DFW_images/DFW_Early_19/19_05_29/DFW_Early_190529_transformed_small.png",hmap_path="/Users/bauera/work/airsurf/wheat/DFW_images/DFW_Early_19/19_05_29/DFW_Early_190529Height_Map_trans.png")
Pipeline.getInstance().run_pipeline(".", parent_dir="/Users/bauera/work/airsurf/wheat/DFW_images/DFW_Early_19", seg_path="/Users/bauera/work/airsurf/wheat/DFW_images/DFW_Early_19/19_06_05")
Pipeline.getInstance().run_pipeline(".", parent_dir="/Users/bauera/work/airsurf/wheat/DFW_images/DFW_Mid_19", seg_path="/Users/bauera/work/airsurf/wheat/DFW_images/DFW_Mid_19/19_05_29")
Pipeline.getInstance().run_pipeline(".", parent_dir="/Users/bauera/work/airsurf/wheat/DFW_images/DFW_Late_19", seg_path="/Users/bauera/work/airsurf/wheat/DFW_images/DFW_Late_19/19_05_29")
# Interface.getInstance().run()
