import baostock as bs
import pandas as pd
import datetime

OUTPUT = './stockdata'


def download_all_stock_day_k(date):
    bs.login()

    # 获取指定日期的指数、股票数据
    stock_rs = bs.query_all_stock(date)
    stock_df = stock_rs.get_data()
    data_df = pd.DataFrame()
    for code in stock_df["code"]:
        print("Downloading :" + code)
        k_rs = bs.query_history_k_data_plus(code, "date,code,open,high,low,close", date, date)
        data_df = data_df.append(k_rs.get_data())
    bs.logout()
    data_df.to_csv(f"{OUTPUT}/{date}.csv", encoding="gbk", index=False)


class Downloader(object):
    def __init__(self):
        self._bs = bs
        bs.login()
        self.date_start = '1990-01-01'
        # self.date_end = datetime.datetime.now().strftime("%Y-%m-%d")
        self.date_end = '2020-03-23'
        self.fields = "date,code,open,high,low,close,volume,amount," \
                      "adjustflag,turn,tradestatus,pctChg,peTTM," \
                      "pbMRQ,psTTM,pcfNcfTTM,isST"

    def exit(self):
        bs.logout()

    def get_codes_by_date(self, date):
        print(date)
        stock_rs = bs.query_all_stock(date)
        stock_df = stock_rs.get_data()
        print(stock_df)
        return stock_df["code"]

    def run(self):
        codes = self.get_codes_by_date(self.date_end)
        for code in codes:
            print(f'processing {code}')
            df_code = bs.query_history_k_data_plus(code, self.fields,
                                                   start_date=self.date_start,
                                                   end_date=self.date_end).get_data()
            df_code.to_csv(f'{OUTPUT}/{code}.csv', index=False)
        self.exit()


if __name__ == '__main__':
    # 获取指定日期全部股票的日K线数据
    downloader = Downloader()
    downloader.run()
    # downloader.get_codes_by_date('2020-03-23')
