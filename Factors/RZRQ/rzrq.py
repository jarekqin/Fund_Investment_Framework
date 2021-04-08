import os


class RZRQDailyTask:

    _bash_file_name='rzrq_to_sql.bat'
    _shell_file_name='SZEX_RZRQ.py'
    _citi_file_name='CITI_Industry_Analysis.py'

    @classmethod
    def main(cls):
        # 先执行抓取csv文件shell脚本
        if os.system(os.path.abspath(cls._shell_file_name))==0:
            if os.system(os.path.abspath(cls._bash_file_name))==0:
                print('rzrq每日相关任务执行完成!')
            else:
                print('rzrq的bat脚本无法执行')
        else:
            raise NotImplementedError('shell脚本无法执行')

        if os.system(os.path.abspath(cls._citi_file_name))==0:
            print('两融可视化脚本执行结束!')
        else:
            raise NotImplementedError('两融可视化执行失败!')

if __name__=='__main__':

    rzrq = RZRQDailyTask()
    RZRQDailyTask.main()