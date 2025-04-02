from extractVoice import call_llama_api
from generate import generateToImg
import json

def generate_storyboard_images(prompt_text):
    # 1. 获取storyboard数据
    response = call_llama_api(prompt_text)
    import re
    json_match = re.search(r'\{[\s\S]*\}', response.choices[0].message.content)
    if not json_match:
        raise ValueError("未找到有效的JSON内容")
    storyboard = json.loads(json_match.group(0))
    
    # 2. 按scene顺序生成图片
    for scene in storyboard['storyboard']:
        print(f"正在生成场景 {scene['scene_id']}: {scene['description']}")
        
        # 3. 调用generateToImg
        has_main_character = "主角" in scene['description'] and "林云" in scene['description']
        try:
            print(f"\n开始生成场景 {scene['scene_id']}")
            generator = generateToImg(
                adapter_name=f"scene_{scene['scene_id']}",
                charac_name="林云" if has_main_character else "",  # 角色名
                prompt=scene['prompt'],
                negative_prompt=scene['negative_prompt'],
                fix_prompt=None,  # 不使用fix_prompt
                charac=has_main_character  # 根据描述判断是否使用角色形象
            )
            saved_files = generator.generate()
            print(f"场景 {scene['scene_id']} 生成完成: {saved_files}")
        except Exception as e:
            print(f"场景 {scene['scene_id']} 生成失败: {str(e)}")
            continue
        
        print(f"场景 {scene['scene_id']} 图片生成完成")

if __name__ == "__main__":
    prompt = """
    赵院士缓缓的说道。
 吴院士也在一旁认同的点了点头，从现在的陆战之王坦克身上就能看出这一点来，面对导弹以及飞机，坦克只是一个会跑的靶子。
 张远航面带微笑的看着赵院士和吴院士两人，自己的机甲可不是一个靶子，俗话说的好最好的防御就是主动出击。
 要知道机甲上的武器可不少了，就比如说隐藏在机甲肩膀下的激光炮。
 只不过张远航并不打算现在说出来，先让大家消化一下，免得等下太激动，倒下一两个人那就不好了。
 李兵院士此时注意到了张远航听到赵院士和吴院士的话后，眼底浮现了一丝自信。
 难道这个孩子的机甲还有着什么特别之处嘛？？？
 这孩子是上天给我们华夏最宝贵的礼物啊！！！
 这时张远航也注意到机甲驾驶室里的林云已经快到达极限了，立马让机甲停止下来，然后下降。
 接到张远航的命令后，机甲缓缓减速，慢慢回到了刚刚起飞的位置。
 “砰！”
 当机甲落在地面后，林云第一时间打开了驾驶室，跌跌撞撞的从上面爬了下来。
 “呕！”
 刚刚踩在地上，林云就开始吐了起来。
    """
    generate_storyboard_images(prompt)
