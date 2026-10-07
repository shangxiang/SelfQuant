"""微信推送（PushPlus）。"""
import requests


def send_wechat(title: str, content: str, token: str = "cd742d4cdb3144dd89b82fae099edb11") -> bool:
    """
    通过 PushPlus 推送消息到微信。

    Parameters
    ----------
    title : str
        消息标题
    content : str
        消息内容（支持 Markdown）
    token : str, optional
        PushPlus token，不传则从环境变量 PUSHPLUS_TOKEN 读取

    Returns
    -------
    bool
        推送是否成功
    """
    if token is None:
        import os
        token = os.getenv("PUSHPLUS_TOKEN")
        if not token:
            print("[PushPlus] 未配置 token，跳过推送")
            return False

    url = "http://www.pushplus.plus/send"
    payload = {
        "token": token,
        "title": title,
        "content": content,
        "template": "markdown",
    }

    try:
        resp = requests.post(url, json=payload, timeout=10)
        data = resp.json()
        if data.get("code") == 200:
            print(f"[PushPlus] 推送成功: {title}")
            return True
        else:
            print(f"[PushPlus] 推送失败: {data.get('msg')}")
            return False
    except Exception as e:
        print(f"[PushPlus] 推送异常: {e}")
        return False


if __name__ == "__main__":
    # 测试推送
    send_wechat(
        title="测试消息",
        content="## SelfQuant 测试\n\n这是一条测试消息\n- 买入：贵州茅台\n- 卖出：宁德时代"
    )
