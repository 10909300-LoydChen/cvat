// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React, { useCallback, useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import Alert from 'antd/lib/alert';
import Button from 'antd/lib/button';
import Drawer from 'antd/lib/drawer';
import Progress from 'antd/lib/progress';
import Tag from 'antd/lib/tag';
import Text from 'antd/lib/typography/Text';
import Title from 'antd/lib/typography/Title';
import { ExperimentOutlined } from '@ant-design/icons';

import { CombinedState } from 'reducers';
import { shallowEqual, ThunkDispatch } from 'utils/redux';
import { changeFrameAsync } from 'actions/annotation-actions';
import {
    ModelStatus, controlTraining, fetchModelStatus, loadJobScores, nextSuggestedFrame, scoreJob, useJobScores,
} from 'utils/active-learning';

const STATUS_POLL_MS = 3000;
const RANKED_FRAMES = 20;

function trainingTag(status: ModelStatus): JSX.Element {
    if (status.training) return <Tag color='processing'>Training</Tag>;
    if (status.full_round_requested) return <Tag color='processing'>Full round queued</Tag>;
    if (status.paused) return <Tag color='warning'>Paused</Tag>;
    return <Tag color='success'>Idle</Tag>;
}

function ActiveLearningControl(): JSX.Element {
    const dispatch = useDispatch<ThunkDispatch>();
    const { jobID, currentFrame, org } = useSelector((state: CombinedState) => ({
        jobID: state.annotation.job.instance?.id,
        currentFrame: state.annotation.player.frame.number,
        org: state.organizations.current?.slug ?? '',
    }), shallowEqual);
    const [open, setOpen] = useState(false);
    const [status, setStatus] = useState<ModelStatus | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [busy, setBusy] = useState(false);
    const scores = useJobScores(jobID);

    const refresh = useCallback((): void => {
        fetchModelStatus(org).then((value) => {
            setStatus(value);
            setError(null);
        }).catch((e: Error) => setError(e.message));
    }, [org]);

    useEffect(() => {
        if (!open) return undefined;
        refresh();
        const timer = setInterval(refresh, STATUS_POLL_MS);
        return () => clearInterval(timer);
    }, [open, refresh]);

    useEffect(() => {
        if (open && typeof jobID === 'number' && !scores) {
            loadJobScores(jobID, org).catch(() => {});
        }
    }, [open, jobID]);

    const act = (action: 'pause' | 'resume' | 'train'): void => {
        setBusy(true);
        controlTraining(action, org)
            .then(setStatus)
            .catch((e: Error) => setError(e.message))
            .finally(() => setBusy(false));
    };

    const score = (): void => {
        if (typeof jobID === 'number') {
            scoreJob(jobID, org).catch((e: Error) => setError(e.message));
        }
    };

    const goTo = (frame: number): void => {
        if (frame !== currentFrame) {
            dispatch(changeFrameAsync(frame));
        }
    };

    const next = nextSuggestedFrame(scores, currentFrame);
    const ranked = [...(scores?.frames ?? [])].sort((a, b) => b.score - a.score).slice(0, RANKED_FRAMES);
    const running = scores?.state === 'running';

    return (
        <>
            <Button
                type='link'
                className='cvat-annotation-header-model-button cvat-annotation-header-button'
                onClick={() => setOpen(true)}
            >
                <ExperimentOutlined />
                Model
            </Button>
            <Drawer
                title='Model and active learning'
                open={open}
                onClose={() => setOpen(false)}
                width={420}
                className='cvat-active-learning-drawer'
            >
                {error && <Alert type='error' showIcon message={error} className='cvat-active-learning-error' />}

                <section className='cvat-active-learning-section'>
                    <Title level={5}>Model</Title>
                    {status ? (
                        <>
                            <div className='cvat-active-learning-row'>
                                <Text type='secondary'>State</Text>
                                {trainingTag(status)}
                            </div>
                            <div className='cvat-active-learning-row'>
                                <Text type='secondary'>Version</Text>
                                <Text strong>
                                    {`v${status.version}`}
                                    {status.metrics.round ? ` (${status.metrics.round} round)` : ''}
                                </Text>
                            </div>
                            <div className='cvat-active-learning-row'>
                                <Text type='secondary'>Validation mAP50</Text>
                                <Text strong>
                                    {typeof status.metrics.val_map50 === 'number' ? (
                                        `${status.metrics.val_map50.toFixed(3)} ` +
                                        `on ${status.metrics.val_samples} frames`
                                    ) : 'not measured yet'}
                                </Text>
                            </div>
                            <div className='cvat-active-learning-row'>
                                <Text type='secondary'>Training samples</Text>
                                <Text strong>
                                    {`${status.samples}`}
                                    {status.pending_new_samples ? `, ${status.pending_new_samples} not learned yet` : ''}
                                </Text>
                            </div>
                            <div className='cvat-active-learning-row'>
                                <Text type='secondary'>Labels</Text>
                                <span>{status.labels.map((label) => <Tag key={label}>{label}</Tag>)}</span>
                            </div>
                            <div className='cvat-active-learning-actions'>
                                <Button
                                    loading={busy}
                                    onClick={() => act(status.paused ? 'resume' : 'pause')}
                                >
                                    {status.paused ? 'Resume automatic training' : 'Pause automatic training'}
                                </Button>
                                <Button
                                    type='primary'
                                    loading={busy}
                                    disabled={status.training || status.full_round_requested}
                                    onClick={() => act('train')}
                                >
                                    Train now (full round)
                                </Button>
                            </div>
                            <Text type='secondary' className='cvat-active-learning-hint'>
                                Pausing stops rounds started by saves. Annotations are still collected and
                                learned after resuming. A full round started here also runs while paused.
                            </Text>
                        </>
                    ) : !error && <Text type='secondary'>Loading…</Text>}
                </section>

                <section className='cvat-active-learning-section'>
                    <Title level={5}>Active learning</Title>
                    <Text type='secondary' className='cvat-active-learning-hint'>
                        Scores every frame of this job with the current model. The frames where the model is
                        least sure come first, and the frame strip shows the scores.
                    </Text>
                    <div className='cvat-active-learning-actions'>
                        <Button onClick={score} loading={running} disabled={typeof jobID !== 'number'}>
                            {scores && scores.state !== 'idle' ? 'Score frames again' : 'Score frames'}
                        </Button>
                        <Button type='primary' disabled={!next} onClick={() => next && goTo(next.frame)}>
                            {next ? `Next suggested frame (${next.frame})` : 'Next suggested frame'}
                        </Button>
                    </div>
                    {running && scores && (
                        <Progress percent={Math.round((100 * scores.progress) / Math.max(scores.total, 1))} />
                    )}
                    {scores?.state === 'failed' && <Alert type='error' showIcon message={scores.error} />}
                    {scores?.state === 'done' && status && scores.model_version !== status.version && (
                        <Alert type='warning' showIcon message='The model changed since scoring. Score again to refresh.' />
                    )}
                    {scores?.state === 'done' && scores.model_version === 0 && (
                        <Alert type='warning' showIcon message='The model is not trained yet, so the scores are not meaningful.' />
                    )}
                    {ranked.length > 0 && (
                        <ol className='cvat-active-learning-ranking'>
                            {ranked.map((item) => (
                                <li key={item.frame}>
                                    <button
                                        type='button'
                                        className={item.frame === currentFrame ? 'cvat-active-learning-current' : ''}
                                        onClick={() => goTo(item.frame)}
                                    >
                                        <Text strong>{`Frame ${item.frame}`}</Text>
                                        <Progress
                                            percent={Math.round(item.score * 100)}
                                            size='small'
                                            showInfo={false}
                                            strokeColor='#fa8c16'
                                        />
                                        <Text type='secondary'>{`${Math.round(item.score * 100)}`}</Text>
                                        {item.annotated && <Tag>annotated</Tag>}
                                    </button>
                                </li>
                            ))}
                        </ol>
                    )}
                </section>
            </Drawer>
        </>
    );
}

export default React.memo(ActiveLearningControl);
