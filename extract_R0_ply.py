import os
import re
import zipfile
import shutil
from pathlib import Path
from typing import Dict

IN_ZIP = "data/SparsePCGC_dense_lossy_1023.zip"   # 输入：原始数据包
TMP_UNPACK = "_tmp_unpack_sparsepcgc"        # 临时解压目录（脚本结束会清理）
OUT_TPL = "ModelNet40-Sparse-dense-lossy1023-r0{lvl}.zip"     # 输出 zip 文件名模板

# ModelNet40 类别（用于识别类别）
MODELNET40_CLASSES = {
    "airplane","bathtub","bed","bench","bookshelf","bottle","bowl","car","chair","cone","cup",
    "curtain","desk","door","dresser","flower_pot","glass_box","guitar","keyboard","lamp","laptop",
    "mantel","monitor","night_stand","person","piano","plant","radio","range_hood","sink","sofa",
    "stairs","stool","table","tent","toilet","tv_stand","vase","wardrobe","xbox"
}

# 正则：匹配 *_Rk.ply（k=0..5），且不是 *_ref.ply
RE_RECON = re.compile(r"_R([0-5])\.ply$", re.IGNORECASE)
RE_REF   = re.compile(r"_ref\.ply$", re.IGNORECASE)

def guess_class(filepath: str, filename: str) -> str | None:
    """优先用文件名 token 推断类别，不行再从目录名里找。"""
    token = filename.split("_")[0].lower()
    if token in MODELNET40_CLASSES:
        return token
    for p in Path(filepath).parts[::-1]:
        p = p.lower()
        if p in MODELNET40_CLASSES:
            return p
    return None

def main():
    if not os.path.exists(IN_ZIP):
        raise FileNotFoundError(f"Input zip not found: {IN_ZIP}")

    # 清理临时目录
    if os.path.exists(TMP_UNPACK):
        shutil.rmtree(TMP_UNPACK)

    # 解压
    print(f"[1/3] Unzipping {IN_ZIP} ...")
    with zipfile.ZipFile(IN_ZIP, "r") as zf:
        zf.extractall(TMP_UNPACK)

    # 预创建 6 个输出 zip 句柄（写模式）
    print("[2/3] Opening 6 output zips ...")
    out_zips: Dict[int, zipfile.ZipFile] = {}
    try:
        for lvl in range(6):
            out_name = OUT_TPL.format(lvl=lvl)
            if os.path.exists(out_name):
                os.remove(out_name)
            out_zips[lvl] = zipfile.ZipFile(out_name, "w", compression=zipfile.ZIP_DEFLATED)

        # 遍历临时目录，挑选 *_Rk.ply（非 _ref）
        print("[3/3] Scanning and writing files ...")
        cnt_total = 0
        cnt_per_lvl = {i: 0 for i in range(6)}
        skipped_no_class = 0

        for root, _, files in os.walk(TMP_UNPACK):
            for fn in files:
                if not fn.lower().endswith(".ply"):
                    continue
                if RE_REF.search(fn):
                    # 跳过 *_ref.ply
                    continue

                m = RE_RECON.search(fn)
                if not m:
                    # 非 *_Rk.ply（比如原始或其它），跳过
                    continue

                lvl = int(m.group(1))  # 0..5
                cls = guess_class(os.path.join(root, fn), fn)
                if not cls:
                    skipped_no_class += 1
                    continue

                # 归一化名称：去掉 _Rk 后缀，例如 airplane_0627_R0.ply -> airplane_0627.ply
                stem, ext = os.path.splitext(fn)
                norm_name = re.sub(r"_R[0-5]$", "", stem, flags=re.IGNORECASE) + ext

                # 归档名：<class>/test/<norm_name>
                arcname = f"{cls}/test/{norm_name}"

                # 直接把磁盘文件写入对应 level 的 zip（不创建中间目录）
                out_zips[lvl].write(os.path.join(root, fn), arcname)

                cnt_total += 1
                cnt_per_lvl[lvl] += 1

        # 关闭 zips
        for z in out_zips.values():
            z.close()

        print("Done.")
        print(f"Total files packed: {cnt_total}")
        for lvl in range(6):
            print(f"  R{lvl}: {cnt_per_lvl[lvl]} files -> {OUT_TPL.format(lvl=lvl)}")
        if skipped_no_class:
            print(f"  Skipped (unknown class): {skipped_no_class}")

    finally:
        # 确保关闭
        for lvl, z in list(out_zips.items()):
            try:
                z.close()
            except Exception:
                pass

        # 清理临时目录
        if os.path.exists(TMP_UNPACK):
            shutil.rmtree(TMP_UNPACK)

if __name__ == "__main__":
    main()
