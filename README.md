# 如何用深度强化学习自动炒股

## 💡 初衷

最近一段时间，受到新冠疫情的影响，股市接连下跌，作为一棵小白菜兼小韭菜，竟然产生了抄底的大胆想法，拿出仅存的一点私房钱梭哈了一把。

第二天，暴跌，俺加仓

第三天，又跌，俺加仓

第三天，又跌，俺又加仓...

<img src="img/2020-03-27-10-45-59.png" alt="drawing" width="50%"/>

一番错误操作后，结果惨不忍睹，第一次买股票就被股市一段暴打，受到了媳妇无情的嘲讽。痛定思痛，俺决定换一个思路：**如何用深度强化学习来自动模拟炒股？** 实验验证一下能否获得收益。

## 📖 监督学习与强化学习的区别

监督学习（如 LSTM）可以根据各种历史数据来预测未来的股票的价格，判断股票是涨还是跌，帮助人做决策。

<img src="img/2020-03-25-18-55-13.png" alt="drawing" width="50%"/>

而强化学习是机器学习的另一个分支，在决策的时候采取合适的行动 (Action) 使最后的奖励最大化。与监督学习预测未来的数值不同，强化学习根据输入的状态（如当日开盘价、收盘价等），输出系列动作（例如：买进、持有、卖出），使得最后的收益最大化，实现自动交易。

<img src="img/2020-03-25-18-19-03.png" alt="drawing" width="50%"/>

## 🤖 OpenAI Gym 股票交易环境

### 观测 Observation

策略网络观测的就是一支股票的各项参数，比如开盘价、收盘价、成交数量等。部分数值会是一个很大的数值，比如成交金额或者成交量，有可能百万、千万乃至更大，为了训练时网络收敛，观测的状态数据输入时，必须要进行归一化，变换到 `[-1, 1]` 的区间内。

|参数名称|参数描述|说明|
|---|---|---|
|date|交易所行情日期|格式：YYYY-MM-DD|
|code|证券代码|格式：sh.600000。sh：上海，sz：深圳|
|open|今开盘价格|精度：小数点后4位；单位：人民币元|
|high|最高价|精度：小数点后4位；单位：人民币元|
|low|最低价|精度：小数点后4位；单位：人民币元|
|close|今收盘价|精度：小数点后4位；单位：人民币元|
|preclose|昨日收盘价|精度：小数点后4位；单位：人民币元|
|volume|成交数量|单位：股|
|amount|成交金额|精度：小数点后4位；单位：人民币元|
|adjustflag|复权状态|不复权、前复权、后复权|
|turn|换手率|精度：小数点后6位；单位：%|
|tradestatus|交易状态|1：正常交易 0：停牌|
|pctChg|涨跌幅（百分比）|精度：小数点后6位|
|peTTM|滚动市盈率|精度：小数点后6位|
|psTTM|滚动市销率|精度：小数点后6位|
|pcfNcfTTM|滚动市现率|精度：小数点后6位|
|pbMRQ|市净率|精度：小数点后6位|

### 动作 Action

假设交易共有**买入**、**卖出**和**保持** 3 种操作，定义动作(`action`)为长度为 2 的数组

- `action[0]` 为操作类型；
- `action[1]` 表示买入或卖出百分比；

| 动作类型 `action[0]` | 说明 |
|---|---|
| 1 | 买入 `action[1]`|
| 2 | 卖出 `action[1]`|
| 3 | 保持 |

注意，当动作类型 `action[0] = 3` 时，表示不买也不抛售股票，此时 `action[1]` 的值无实际意义，网络在训练过程中，Agent 会慢慢学习到这一信息。

### 奖励 Reward

奖励函数的设计，对强化学习的目标至关重要。在股票交易的环境下，最应该关心的就是当前的盈利情况，故用当前的利润作为奖励函数。即`当前本金 + 股票价值 - 初始本金 = 利润`。

```python
# profits
reward = self.net_worth - INITIAL_ACCOUNT_BALANCE
reward = 1 if reward > 0 else reward = -100
```

为了使网络更快学习到盈利的策略，当利润为负值时，给予网络一个较大的惩罚 (`-100`)。

## 🕵️‍♀️ 模拟实验

### 环境安装

```sh
# 虚拟环境
virtualenv -p python3.6 venv
source ./venv/bin/activate
# 安装库依赖
pip install -r requirements.txt
```

### 股票数据获取

股票证券数据集来自于 [baostock](http://baostock.com/baostock/index.php/%E9%A6%96%E9%A1%B5)，一个免费、开源的证券数据平台，提供 Python API。

```bash
>> pip install baostock -i https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn
```

数据获取代码参考 [get_stock_data.py](https://github.com/wangshub/RL-Stock/blob/master/get_data.py)

```python
>> python get_stock_data.py
```

将过去 20 多年的股票数据划分为训练集，和末尾 1 个月数据作为测试集，来验证强化学习策略的有效性。划分如下

| `1990-01-01` ~ `2019-11-29` | `2019-12-01` ~ `2019-12-31` |
|---|---|
| 训练集 | 测试集 |

### 验证结果

**单只股票**

- 初始本金 `10000`
- 股票代码：`sh.600036`(招商银行)
- 训练集： `stockdata/train/sh.600036.招商银行.csv`
- 测试集： `stockdata/test/sh.600036.招商银行.csv`
- 模拟操作 `20` 天，最终盈利约 `400`

<img src="img/sh.600036.png" alt="drawing" width="70%"/>

**多只股票**

选取 `79` 只股票，进行训练，共计 

- 盈利： `44.8%`
- 不亏不赚： `49.0%`
- 亏损：`6.3%`

<img src="img/pie.png" alt="drawing" width="60%"/>

<img src="img/hist.png" alt="drawing" width="60%"/>

## 👻 最后

- 俺完全是股票没入门的新手，难免存在错误，欢迎指正！
- 数据和方法皆来源于网络，无法保证有效性，**Just For Fun**！

## 📚 参考资料

- Y. Deng, F. Bao, Y. Kong, Z. Ren and Q. Dai, "Deep Direct Reinforcement Learning for Financial Signal Representation and Trading," in IEEE Transactions on Neural Networks and Learning Systems, vol. 28, no. 3, pp. 653-664, March 2017.
- [Yuqin Dai, Chris Wang, Iris Wang, Yilun Xu, "Reinforcement Learning for FX trading"](http://stanford.edu/class/msande448/2019/Final_reports/gr2.pdf)
- [Create custom gym environments from scratch — A stock market example](https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e)
- [notadamking/Stock-Trading-Environment](https://github.com/notadamking/Stock-Trading-Environment)
- Chien Yi Huang. Financial trading as a game: A deep reinforcement learning approach. arXiv preprint arXiv:1807.02787, 2018.
