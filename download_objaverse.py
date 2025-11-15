import objaverse

# 用 include_textures=True 强制拉下贴图
uids = ["329684e1e63d4d16bcf519cc9571c1fb"]
paths = objaverse.load_objects(
    uids,
    download_processes=4,
    include_textures=True
)
print(paths)
# -> {'329684e1e63d4d16bcf519cc9571c1fb': '/home/.../.objaverse/glbs/000-011/329684e1e63d4d16bcf519cc9571c1fb.glb'}
