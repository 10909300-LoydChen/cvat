// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React, { useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import Button from 'antd/lib/button';
import Modal from 'antd/lib/modal';
import Spin from 'antd/lib/spin';
import { EditOutlined } from '@ant-design/icons';

import {
    getCore, Job, Project, Task,
} from 'cvat-core-wrapper';
import { CombinedState } from 'reducers';
import { getCVATStore } from 'cvat-store';
import { ThunkDispatch } from 'utils/redux';
import { getJobAsync, saveAnnotationsAsync } from 'actions/annotation-actions';
import { updateTaskAsync } from 'actions/tasks-actions';
import { updateProjectAsync } from 'actions/projects-actions';
import CVATTooltip from 'components/common/cvat-tooltip';
import LabelsEditor from 'components/labels-editor/labels-editor';

const core = getCore();

// Labels are owned by the project (if any) or the task, not by the job
async function fetchLabelsOwner(job: Job): Promise<Task | Project> {
    if (job.projectId) {
        const [project] = await core.projects.get({ id: job.projectId });
        return project;
    }

    const [task] = await core.tasks.get({ id: job.taskId });
    return task;
}

function LabelsEditorModal(): JSX.Element | null {
    const dispatch = useDispatch<ThunkDispatch>();
    const job = useSelector((state: CombinedState) => state.annotation.job.instance);
    const [owner, setOwner] = useState<Task | Project | null>(null);
    const [visible, setVisible] = useState(false);
    const [changed, setChanged] = useState(false);

    if (!job) {
        return null;
    }

    const onOpen = async (): Promise<void> => {
        setVisible(true);
        try {
            // labels deletion removes related annotations on the server, so local changes must be saved first
            if (job.annotations.hasUnsavedChanges()) {
                await dispatch(saveAnnotationsAsync());
            }
            setOwner(await fetchLabelsOwner(job));
        } catch (error: unknown) {
            setVisible(false);
            Modal.error({ title: 'Could not open labels editor', content: (error as Error).message });
        }
    };

    const onSubmit = async (labels: any[]): Promise<void> => {
        const fields = { labels: labels.map((labelData): any => new core.classes.Label(labelData)) };
        const updated = owner instanceof core.classes.Project ?
            await dispatch(updateProjectAsync(owner as Project, fields)) :
            await dispatch(updateTaskAsync(owner as Task, fields));
        setOwner(updated);
        setChanged(true);
    };

    const onClose = (): void => {
        setVisible(false);
        setOwner(null);
        if (changed) {
            // job and annotations collection keep labels from initialization, so reload them
            setChanged(false);
            const { annotation } = getCVATStore().getState();
            dispatch(getJobAsync({
                taskID: job.taskId,
                jobID: job.id,
                initialFrame: annotation.player.frame.number,
                initialFilters: annotation.annotations.filters,
                queryParameters: {
                    ...annotation.job.queryParameters,
                    initialOpenGuide: false,
                    initialWorkspace: annotation.workspace,
                },
            }));
        }
    };

    return (
        <>
            <CVATTooltip title='Add or edit labels'>
                <Button
                    className='cvat-objects-sidebar-labels-edit-button'
                    type='link'
                    size='small'
                    icon={<EditOutlined />}
                    onClick={onOpen}
                />
            </CVATTooltip>
            <Modal
                className='cvat-annotation-labels-editor-modal'
                title={owner instanceof core.classes.Project ? 'Project labels' : 'Task labels'}
                open={visible}
                onCancel={onClose}
                footer={null}
                width={800}
                destroyOnClose
            >
                {owner ? (
                    <LabelsEditor
                        labels={owner.labels.map((label) => label.toJSON())}
                        onSubmit={onSubmit}
                    />
                ) : <Spin />}
            </Modal>
        </>
    );
}

export default React.memo(LabelsEditorModal);
