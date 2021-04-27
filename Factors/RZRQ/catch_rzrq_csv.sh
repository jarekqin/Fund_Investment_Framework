#！/bin/bash
url="http://www.szse.cn/api/report/ShowReport?SHOWTYPE=xlsx&CATALOGID=1837_xxpl&txtDate="
url2="http://www.sse.com.cn/market/dealingdata/overview/margin/a/rzrqjygk"

url_end="&random=0.14476487458089227&TABKEY=tab2"
url_end2=".xls"
file_end="sz_rzrq_data.csv"
file_end2="sh_rzrq_data.xls"


WEEK_DAY=$(date +%w)

if (($WEEK_DAY == 1));
then
  startDay=$(date -d "now -3day" +%Y-%m-%d)
  startDay=$(date -d "$startDay" +%s)
  endDay=$(date -d "now -3 day" +%Y-%m-%d)
  endDay=$(date -d "now -3 day" +%s)
else
  startDay=$(date -d "now -1 day" +%Y-%m-%d)
  startDay=$(date -d "$startDay" +%s)
  endDay=$(date -d "now -1 day" +%Y-%m-%d)
  endDay=$(date -d "now -1 day" +%s)
fi

#startDay=$(date -d "2021-04-22" +%Y-%m-%d)
#startDay=$(date -d "$startDay" +%s)
#endDay=$(date -d "2021-04-22" +%Y-%m-%d)
#endDay=$(date -d "$endDay" +%s)

root="H:\\RZRQ_csv\\sz\\"
root2="H:\\RZRQ_csv\\sh\\"

while (($startDay <= $endDay)); do
  record_date=$(date -d @$startDay "+%Y-%m-%d")
  record_date2=$(date -d @$startDay "+%Y%m%d")
  wget ${url}${record_date}${url_end} -O ${root}${record_date}${file_end}
  wget ${url2}${record_date2}${url_end2} -O ${root2}${record_date}${file_end2}
#  echo ${url}${record_date}${url_end} -O ${root}${record_date}${file_end}
  startDay=$(date -d @$startDay "+%Y-%m-%d")
  startDay=$(date -d "$startDay +1 day " +%Y-%m-%d)
  startDay=$(date -d "$startDay" +%s)
  sleep 1
#
done


#http://www.szse.cn/api/report/ShowReport?SHOWTYPE=xlsx&CATALOGID=1837_xxpl&txtDate=2020-09-02&random=0.14476487458089227&TABKEY=tab2