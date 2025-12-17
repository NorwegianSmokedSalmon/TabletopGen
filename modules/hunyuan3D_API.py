from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.ai3d.v20250513 import ai3d_client, models

import requests
import time
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
import json
import os

# Temporarily disable proxy settings
_proxy_keys = ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]
_old_proxy = {k: os.environ.get(k) for k in _proxy_keys}



def to_base64(path):
    import base64
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

image_path = "path/to/your/image.jpg" 

def submit_job(client, image_path):
    # Construct request
    req = models.SubmitHunyuanTo3DProJobRequest()
    
    params = {
        "ImageBase64": to_base64(image_path),
        "GenerateType": "Normal",  # Normal/LowPoly/Geometry/Sketch
        "EnablePBR": True,  # Optional: Enable PBR material
        # "FaceCount": 500000,  # Optional: Model face count
    }
    req.from_json_string(json.dumps(params))

    # Call interface
    resp = client.SubmitHunyuanTo3DProJob(req)
    job_id = resp.JobId
    print(f"Job submitted, JobID: {job_id}")
    return job_id

def query_job(client, job_id):
    req = models.QueryHunyuanTo3DProJobRequest()
    req.JobId = job_id
    return client.QueryHunyuanTo3DProJob(req)


def gen_single_obj_hy3dapi(SecretId, SecretKey, image_path, output_glb_path):
    """
    Step 3 核心功能：调用腾讯云 Hunyuan3D API 将单张 2D 图像生成为 3D 模型 (GLB)。

    功能描述：
        1. 初始化腾讯云 AI3D 客户端。
        2. 读取本地图片并转换为 Base64 编码。
        3. 构造 SubmitHunyuanTo3DProJobRequest 请求，提交 3D 生成任务。
        4. 轮询 QueryHunyuanTo3DProJob 接口检查任务状态。
        5. 任务完成后，从返回的 ResultFile3Ds 中获取 GLB 下载链接。
        6. 下载 GLB 文件并保存到 output_glb_path。

    输入 (Input):
        - SecretId (str): 腾讯云 API 密钥 ID。
        - SecretKey (str): 腾讯云 API 密钥 Key。
        - image_path (str): 输入的 2D 图像路径（通常是 Step 3a 重绘后的物体图像）。
        - output_glb_path (str): 输出的 3D 模型文件保存路径 (.glb)。

    输出 (Output):
        - 无返回值，但在磁盘上生成 .glb 文件。
        - 如果失败，抛出 RuntimeError 或 TencentCloudSDKException。

    数据结构 (Data Structures):
        - 请求参数 (params):
            {
                "ImageBase64": "base64_string...",  # 必填，图像内容
                "GenerateType": "Normal",           # 生成类型：Normal/LowPoly/Geometry/Sketch
                "EnablePBR": True                   # 是否启用 PBR 材质
            }
        - 响应结构 (response object 'r'):
            r.Status (str): 任务状态 ("WAIT", "RUN", "FAIL", "DONE")
            r.ResultFile3Ds (list): 生成结果列表
                [
                    {
                        "Url": "https://...", # 下载链接
                        "Type": "GLB"         # 文件类型
                    },
                    ...
                ]
    """
    # Configure secret key and region
    cred = credential.Credential(SecretId, SecretKey)  # Replace with actual secret key
    http_profile = HttpProfile(endpoint="ai3d.tencentcloudapi.com")
    client_profile = ClientProfile(httpProfile=http_profile)
    client = ai3d_client.Ai3dClient(cred, "ap-guangzhou", client_profile)


    try:
        job_id = submit_job(client, image_path)
        print("Submission successful, JobId:", job_id)

        # Poll until completion
        while True:
            r = query_job(client, job_id)
            status = r.Status  # WAIT / RUN / FAIL / DONE
            print("Status:", status)
            if status == "DONE":
                # Get GLB from the returned 3D file list
                glb_url = None
                if r.ResultFile3Ds:
                    for f in r.ResultFile3Ds:
                        if (getattr(f, "Type", "") or "").upper() == "GLB":
                            glb_url = f.Url
                            break
                    # If no explicit Type, take the first one directly
                    if not glb_url:
                        glb_url = r.ResultFile3Ds[0].Url
                if not glb_url:
                    raise RuntimeError("Task completed but no GLB link returned.")

                # Download and save
                data = requests.get(glb_url, timeout=120).content
                with open(output_glb_path, "wb") as f:
                    f.write(data)
                print("Saved to:", output_glb_path)
                break

            if status == "FAIL":
                msg = getattr(r, "ErrorMessage", "Unknown error")
                code = getattr(r, "ErrorCode", "")
                raise RuntimeError(f"Task failed: {code} {msg}")

            time.sleep(15)  # Polling

    except TencentCloudSDKException as e:
        print("SDK Exception:", str(e))


if __name__ == "__main__":
    SecretId=""
    SecretKey=""
