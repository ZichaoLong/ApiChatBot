"""
流式响应实时显示处理工具

提供实时显示处理器，用于格式化输出思考过程和最终回答。
"""

from dataclasses import dataclass


@dataclass
class RealTimeDisplayHandler:
    """
    实时显示处理器

    跟踪思考过程和回答的显示状态，添加格式化的标题和分隔符。

    属性:
        first_thoughts_shown: 是否已显示思考过程标题
        first_answer_shown: 是否已显示回答标题
    """

    # 实时显示状态变量
    first_thoughts_shown: bool = False
    first_answer_shown: bool = False

    def _handle_realtime_display(self, text: str, is_thinking: bool, show_thinking: bool) -> None:
        """
        处理第一个candidate的实时显示

        根据内容类型（思考/回答）添加相应的标题和分隔符。

        参数:
            text: 要显示的文本内容
            is_thinking: 是否为思考过程内容
            show_thinking: 是否显示思考过程（配置项）

        显示格式:
            - 思考过程：添加 "🤔 思考过程:" 标题和分隔符
            - 回答内容：添加 "💡 回答:" 标题
        """
        if is_thinking and show_thinking:
            # 首次显示思考内容时，添加标题和分隔符
            if not self.first_thoughts_shown:
                print("\n🤔 思考过程:")
                print("-" * 50)
                self.first_thoughts_shown = True
            print(text, end='', flush=True)
        elif not is_thinking:
            # 首次显示回答内容时，添加标题
            if not self.first_answer_shown and self.first_thoughts_shown:
                # 如果之前显示了思考过程，添加分隔符
                print("\n" + "=" * 50)
                print("💡 回答:")
                self.first_answer_shown = True
            elif not self.first_answer_shown:
                # 如果没有思考过程，直接显示回答标题
                print("💡 回答:")
                self.first_answer_shown = True
            print(text, end='', flush=True)


