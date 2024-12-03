# 相关文件说明
cd path_to_DexGraspNet/grasp_generation
## 可视化机械手
shadow hand: 
    python ./tests/visualize_hand_model.py
kable hand: 
    python ./tests/visualize_kable.py
## 生成数据集
shadow hand: 
    python ./scripts/generate_grasps.py
kable hand: (删除./data/kabledata文件夹之后运行)
    python ./scripts/generate_grasps_kable.py
## 可视化数据集
shadow hand: 
    python ./quick_example.py
kable hand: 
    python ./quick_example_kable.py