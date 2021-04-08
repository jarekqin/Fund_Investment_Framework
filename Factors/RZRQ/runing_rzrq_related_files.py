import os
from omk.interface import AbstractJob


class RZRQDailyTask(AbstractJob):
    _py_file_name = 'SZEX_RZRQ.py'
    _py_file_name2 = 'CITI_Industry_Analysis.py'
    _shell_file_name = 'catch_rzrq_csv.sh'

    @classmethod
    def main(cls):
        # 先执行抓取csv文件shell脚本
        if os.system(os.path.abspath(cls._shell_file_name)) == 0:
            if os.system(r'D:\\anaconda\\envs\\nlp37\\python.exe %s' % os.path.abspath(cls._py_file_name)) == 0:
                if os.system(r'D:\\anaconda\\envs\\nlp37\\python.exe %s' % os.path.abspath(cls._py_file_name2)) == 0:
                    print('rzrq每日相关任务执行完成!')
            else:
                print('rzrq的bat脚本无法执行')
        else:
            raise NotImplementedError('shell脚本无法执行')


if __name__ == '__main__':
    # from jarvis.jobs.runing_rzrq_related_files import RZRQDailyTask
    # from omk.utils.const import JobType
    # JobManager.install_job('rzrq_daily_task', RZRQDailyTask, JobType.Module, activate=True)
    # RQData.init()
    # rzrq = RZRQDailyTask()
    # rzrq_manager = JobManager(alert=True)
    # rzrq.register_event(event_bus=rzrq_manager.event_bus, job_uuid=None, debug=False)
    # rzrq_manager.event_bus.event_queue_reload(EVENT.AM0900)
    # rzrq_manager.event_bus.sequential_publish()
    RZRQDailyTask.main()
