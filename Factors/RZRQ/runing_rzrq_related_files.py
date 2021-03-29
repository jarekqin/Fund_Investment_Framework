import os
from omk.interface import AbstractJob
from omk.utils.const import ProcessDocs, ProcessType, EVENT, TIME
from omk.events import Event
from omk.toolkit.job_tool import JobManager
from omk.core.vendor.RQData import RQData


class RZRQDailyTask(AbstractJob):

    _bash_file_name='rzrq_to_sql.bat'
    _shell_file_name= 'catch_rzrq_csv.sh'


    def register_event(self, event_bus, job_uuid, debug):
        if RQData.check_for_trading_date():
            event_bus.add_listener(Event(
                event_type=EVENT.AM0900,
                func=self.main,
                alert=True,
                gap=None,
                p_type=ProcessType.Jarvis,
                des=ProcessDocs.Jarvis_RZRQ,
                job_uuid=job_uuid,
                retry_n=5,
                retry_freq='10m',
            ))

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


if __name__=='__main__':
    # from jarvis.jobs.runing_rzrq_related_files import RZRQDailyTask
    # from omk.utils.const import JobType
    # JobManager.install_job('rzrq_daily_task', RZRQDailyTask, JobType.Module, activate=True)
    RQData.init()
    rzrq = RZRQDailyTask()
    rzrq_manager = JobManager(alert=True)
    rzrq.register_event(event_bus=rzrq_manager.event_bus, job_uuid=None, debug=False)
    rzrq_manager.event_bus.event_queue_reload(EVENT.AM0900)
    rzrq_manager.event_bus.sequential_publish()
    # RZRQDailyTask.main()