// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import React, { useEffect, useRef, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import Text from 'antd/lib/typography/Text';
import { DownOutlined, UpOutlined } from '@ant-design/icons';

import { getCore } from 'cvat-core-wrapper';
import { CombinedState } from 'reducers';
import { shallowEqual, ThunkDispatch } from 'utils/redux';
import { changeFrameAsync } from 'actions/annotation-actions';
import { FrameScore, useJobScores } from 'utils/active-learning';

const core = getCore();

function FrameFilmstrip(): JSX.Element | null {
    const dispatch = useDispatch<ThunkDispatch>();
    const {
        job, frameNumbers, deletedFrames, currentFrame, organization, navigationBlocked,
    } = useSelector((state: CombinedState) => ({
        job: state.annotation.job.instance,
        frameNumbers: state.annotation.job.frameNumbers,
        deletedFrames: state.annotation.job.meta?.deletedFrames ?? {},
        currentFrame: state.annotation.player.frame.number,
        organization: state.organizations.current,
        navigationBlocked: state.annotation.player.navigationBlocked,
    }), shallowEqual);
    const [collapsed, setCollapsed] = useState(false);
    const itemsRef = useRef<HTMLDivElement>(null);
    const scores = useJobScores(job?.id);
    const scoreByFrame = Object.fromEntries(
        (scores?.frames ?? []).map((item: FrameScore) => [item.frame, item]),
    ) as Record<number, FrameScore>;

    useEffect(() => {
        // the canvas layout subtracts the filmstrip height when it computes its own size
        window.dispatchEvent(new Event('resize'));
    }, [collapsed]);

    useEffect(() => {
        itemsRef.current
            ?.querySelector(`[data-frame="${currentFrame}"]`)
            ?.scrollIntoView({ block: 'nearest', inline: 'center' });
    }, [currentFrame, collapsed]);

    if (!job) {
        return null;
    }

    // frames are loaded by the browser only when they scroll into view; the session cookie authorizes them
    const frameURL = (frame: number): string => (
        `${core.config.backendAPI}/jobs/${job.id}/data?type=frame&quality=compressed` +
        `&number=${frame}&org=${organization?.slug ?? ''}`
    );
    const position = frameNumbers.indexOf(currentFrame) + 1;

    return (
        <div className='cvat-frame-filmstrip'>
            <div className='cvat-frame-filmstrip-header'>
                <Text strong>{`Frames ${position} / ${frameNumbers.length}`}</Text>
                <button
                    type='button'
                    className='cvat-frame-filmstrip-toggle'
                    aria-label={collapsed ? 'Show frames' : 'Hide frames'}
                    onClick={() => setCollapsed(!collapsed)}
                >
                    {collapsed ? <UpOutlined /> : <DownOutlined />}
                </button>
            </div>
            {!collapsed && (
                <div className='cvat-frame-filmstrip-items' ref={itemsRef}>
                    {frameNumbers.map((frame: number) => (
                        <button
                            key={frame}
                            type='button'
                            data-frame={frame}
                            disabled={navigationBlocked}
                            className={[
                                'cvat-frame-filmstrip-item',
                                frame === currentFrame ? 'cvat-frame-filmstrip-item-current' : '',
                                deletedFrames[frame] ? 'cvat-frame-filmstrip-item-deleted' : '',
                            ].join(' ')}
                            onClick={() => {
                                if (frame !== currentFrame) {
                                    dispatch(changeFrameAsync(frame));
                                }
                            }}
                        >
                            <img src={frameURL(frame)} loading='lazy' alt={`Frame ${frame}`} />
                            <span>{frame}</span>
                            {scoreByFrame[frame] && (
                                <span
                                    className={[
                                        'cvat-frame-filmstrip-score',
                                        scoreByFrame[frame].annotated ? 'cvat-frame-filmstrip-score-annotated' : '',
                                    ].join(' ')}
                                    title='Active learning score: how unsure the model is about this frame'
                                >
                                    {scoreByFrame[frame].annotated ? '✓' : Math.round(scoreByFrame[frame].score * 100)}
                                </span>
                            )}
                        </button>
                    ))}
                </div>
            )}
        </div>
    );
}

export default React.memo(FrameFilmstrip);
