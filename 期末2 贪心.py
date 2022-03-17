
'''
假设你是一位顺风车司机，车上最初有 capacity 个空座位可以用来载客。由于道路的限制，车只能向一个方向行驶（也就是说，不允许掉头或改变方向，你可以将其想象为一个向量）。

这儿有一份乘客行程计划表 trips[][]，其中 trips[i] = [num_passengers, start_location, end_location] 包含了第 i 组乘客的行程信息：

必须接送的乘客数量；
乘客的上车地点；
以及乘客的下车地点。
这些给出的地点位置是从你的 初始 出发位置向前行驶到这些地点所需的距离（它们一定在你的行驶方向上）。

请你根据给出的行程计划表和车子的座位数，
来判断你的车是否可以顺利完成接送所用乘客的任务
（当且仅当你可以在所有给定的行程中接送所有乘客时，返回 true，否则请返回 false）。

'''

import requests
import time
from bs4 import BeautifulSoup as bs


def common_crawler(url):
    '''
    通用的爬取网页源代码的函数
    :param url: 网页链接
    :return: 网页源代码
    '''
    header = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit'
        '/537.36 (KHTML, like Gecko) Chrome/80.0.3987.132 Safari/537.36'
    }
    while True:  # 一直循环，直到访问站点成功
        try:
            # 以下except都是用来捕获当requests请求出现异常时，
            # 通过捕获然后等待网络情况的变化，以此来保护程序的不间断运行
            resp = requests.get(url, headers=header, timeout=20)
            break
        except requests.exceptions.ConnectionError:  # 网络连接异常
            print('ConnectionError -- please wait 3 seconds')
            time.sleep(3)
        except requests.exceptions.ChunkedEncodingError:
            print('ChunkedEncodingError -- please wait 3 seconds')
            time.sleep(3)
        except:
            print('Unfortunitely -- An Unknow Error Happened, Please wait 3 seconds')
            time.sleep(3)
            time.sleep(3)

    html_data = resp.text.encode(resp.encoding)
    # 使用BeautifulCoup库进行HTML代码的解析。
    # 第一个参数为需要提取数据的html，第二个参数是指定解析器
    soup = bs(html_data, 'html.parser')
    return soup


def find_right_car(low, high, seat_num):
    '''
    :param low: 司机能承受的汽车价格范围的最低价
    :param high: 司机能承受的汽车价格范围的最高价
    :param seat_num: 汽车座位数
    '''
    # 汽车座位一般只分到8座及以上
    if seat_num >= 7:
        seat_num = 7
    url = 'http://price.cheshi.com/select/' + str(low) + '_' + str(high) + '-0-0-0-0-0-0-0-0-0-' + str(seat_num) +'-1-0'
    soup = common_crawler(url)
    page = soup.find('div', class_='page')
    hrefs = page.find_all('a', target='_self')
    if len(hrefs) > 1:
        for href in hrefs:
            page_url = 'http://price.cheshi.com' + href['href']
            print_all_car(page_url)
    else:
        # 爬取汽车信息
        print_all_car(url)


def print_all_car(url):
    '''
    寻找所有满足司机要求的汽车信息的函数
    :param url: 汽车网页链接
    :return: 无，直接输出满足要求的汽车信息
    '''
    soup = common_crawler(url)
    list = soup.find('div', class_='list')
    # http://price.cheshi.com/series_129.html  战旗
    if list.find_all('dl'):
        dls = list.find_all('dl')
        for dl in dls:
            car_name = dl.find('h3').text
            car_url = ' http://price.cheshi.com' + dl.find('h3').find('a')['href']
            car_price = dl.find_all('p')[1].text
            print(car_name, car_url, car_price)
    else:  # 没有满足条件的汽车信息
        print('None')


def carPooling(trips, capacity, map):
    '''
    :param trips: 顺风车行程订单
    :param capacity: 司机汽车座位数
    :param map: 司机回家路线图
    :return: 如果车座数满足接到全部顺风车订单的要求，就返回车座数，否则返回False
    '''
    travel = {}
    for i in trips:
        if i[1] in travel:
            travel[map[i[1]]] += i[0]
        else:
            travel[map[i[1]]] = i[0]
        if i[2] in travel:
            travel[map[i[2]]] -= i[0]
        else:
            travel[map[i[2]]] = -i[0]
    l = sorted(travel)
    people_in_car = 0
    for i in l:
        people_in_car += travel[i]
        if people_in_car > capacity:
            # print('根据给出的行程计划表和车子的座位数，在初步排除掉不顺路的订单，剩下的订单不能全部完成')
            return False
    print('根据给出的行程计划表和车子的座位数{}，在初步排除掉不顺路的订单，剩下的订单可以全部完成。'.format(capacity))
    return True


if __name__ == '__main__':
    # 司机下班回家路线  地名+处在路线中的顺序
    map = {
            '迪荡新城': 1,  # （司机下班地点）
            '世贸广场': 2,
            '蔡元培故居': 3,
            '绍兴剧院': 4,
            '塔山文化广场': 5,
            '鲁迅小学': 6,
            '绍兴文理学院附中': 7,
            '咸亨大酒店': 8,
            '银泰百货': 9,
            '金时代广场': 10,
            '天马大厦': 11,
            '绍兴洪亮大厦': 12,
            '铭康大厦': 13,
            '绍兴鲁迅高级中学': 14,
            '中成新村': 15,
            '绍兴树人小学（东校区）': 16,
            '缇香名邸': 17,
            '繁荣公寓': 18,
            '浙江天恩太阳能科技有限公司': 19,
            '绍兴文理学院附属医院': 20,
            '南池鉴园': 21,
            '小天才幼儿园': 22,
            '幼山小学': 23,
            '海亮御景园': 24  # （司机家）
            }

    # 顺风车行程订单列表
    #        乘客数  上车地    下车地
    trips = [[2, '世贸广场', '绍兴剧院'],
             [2, '鲁迅小学', '绍兴文理学院附中'],
             [4, '塔山文化广场', '绍兴鲁迅高级中学']]

    flag = False
    for i in range(2, 8):
        capacity = i
        result = carPooling(trips, capacity, map)
        if result:
            low = input('\n司机，请输入你能承受汽车价格范围的最低价。(单位为：万)\n')
            high = input('司机，请输入你能承受的汽车价格范围的最高价。(单位为：万)\n')
            print('\n--------------以下汽车满足司机要求，可选---------------')
            find_right_car(low, high, 9)
            flag = True
            break

    if not flag:
        find_right_car(8, 12, 7)  # 如果座位数为7座还是无法接到所有顺风车订单，那么就选最大的家庭轿车最大的座位数7


