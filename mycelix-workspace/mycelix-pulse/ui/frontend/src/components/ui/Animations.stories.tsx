// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import type { Meta, StoryObj } from '@storybook/react';
import { useState } from 'react';
import {
  Fade,
  Slide,
  Scale,
  Collapse,
  StaggeredList,
  Counter,
  ProgressBar,
  Pulse,
} from './Animations';

const meta: Meta = {
  title: 'UI/Animations',
  tags: ['autodocs'],
  parameters: {
    layout: 'padded',
  },
};

export default meta;

export const FadeAnimation: StoryObj = {
  render: function Render() {
    const [show, setShow] = useState(true);
    return (
      <div className="space-y-4">
        <button
          onClick={() => setShow(!show)}
          className="px-4 py-2 bg-blue-500 text-white rounded-lg"
        >
          Toggle
        </button>
        <Fade show={show}>
          <div className="p-4 bg-blue-100 rounded-lg">
            This content fades in and out
          </div>
        </Fade>
      </div>
    );
  },
};

export const SlideAnimation: StoryObj = {
  render: function Render() {
    const [show, setShow] = useState(true);
    const [direction, setDirection] = useState<'up' | 'down' | 'left' | 'right'>('up');
    return (
      <div className="space-y-4">
        <div className="flex gap-2">
          <button
            onClick={() => setShow(!show)}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg"
          >
            Toggle
          </button>
          <select
            value={direction}
            onChange={(e) => setDirection(e.target.value as any)}
            className="px-4 py-2 border rounded-lg"
          >
            <option value="up">Up</option>
            <option value="down">Down</option>
            <option value="left">Left</option>
            <option value="right">Right</option>
          </select>
        </div>
        <Slide show={show} direction={direction}>
          <div className="p-4 bg-emerald-100 rounded-lg">
            This content slides {direction}
          </div>
        </Slide>
      </div>
    );
  },
};

export const ScaleAnimation: StoryObj = {
  render: function Render() {
    const [show, setShow] = useState(true);
    return (
      <div className="space-y-4">
        <button
          onClick={() => setShow(!show)}
          className="px-4 py-2 bg-blue-500 text-white rounded-lg"
        >
          Toggle
        </button>
        <Scale show={show}>
          <div className="p-4 bg-purple-100 rounded-lg">
            This content scales in and out
          </div>
        </Scale>
      </div>
    );
  },
};

export const CollapseAnimation: StoryObj = {
  render: function Render() {
    const [show, setShow] = useState(true);
    return (
      <div className="space-y-4 w-64">
        <button
          onClick={() => setShow(!show)}
          className="px-4 py-2 bg-blue-500 text-white rounded-lg"
        >
          Toggle
        </button>
        <Collapse show={show}>
          <div className="p-4 bg-amber-100 rounded-lg">
            <p>This content collapses.</p>
            <p className="mt-2">It can have multiple lines.</p>
            <p className="mt-2">The height animates smoothly.</p>
          </div>
        </Collapse>
      </div>
    );
  },
};

export const StaggeredListAnimation: StoryObj = {
  render: function Render() {
    const [show, setShow] = useState(true);
    const items = ['Item 1', 'Item 2', 'Item 3', 'Item 4', 'Item 5'];
    return (
      <div className="space-y-4 w-64">
        <button
          onClick={() => setShow(!show)}
          className="px-4 py-2 bg-blue-500 text-white rounded-lg"
        >
          Toggle
        </button>
        <StaggeredList show={show} staggerDelay={100}>
          {items.map((item) => (
            <div key={item} className="p-3 bg-gray-100 rounded-lg mb-2">
              {item}
            </div>
          ))}
        </StaggeredList>
      </div>
    );
  },
};

export const CounterAnimation: StoryObj = {
  render: function Render() {
    const [value, setValue] = useState(0);
    return (
      <div className="space-y-4 text-center">
        <div className="text-6xl font-bold text-blue-600">
          <Counter value={value} />
        </div>
        <div className="flex gap-2 justify-center">
          <button
            onClick={() => setValue(Math.floor(Math.random() * 1000))}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg"
          >
            Random
          </button>
          <button
            onClick={() => setValue(value + 10)}
            className="px-4 py-2 bg-emerald-500 text-white rounded-lg"
          >
            +10
          </button>
          <button
            onClick={() => setValue(0)}
            className="px-4 py-2 bg-gray-500 text-white rounded-lg"
          >
            Reset
          </button>
        </div>
      </div>
    );
  },
};

export const ProgressBarAnimation: StoryObj = {
  render: function Render() {
    const [value, setValue] = useState(50);
    return (
      <div className="space-y-4 w-64">
        <ProgressBar value={value} />
        <input
          type="range"
          min="0"
          max="100"
          value={value}
          onChange={(e) => setValue(Number(e.target.value))}
          className="w-full"
        />
        <div className="text-center text-sm text-gray-600">{value}%</div>
      </div>
    );
  },
};

export const PulseAnimation: StoryObj = {
  render: function Render() {
    const [active, setActive] = useState(true);
    return (
      <div className="space-y-4">
        <button
          onClick={() => setActive(!active)}
          className="px-4 py-2 bg-blue-500 text-white rounded-lg"
        >
          Toggle Pulse
        </button>
        <div className="flex items-center gap-4">
          <Pulse active={active}>
            <div className="w-4 h-4 bg-red-500 rounded-full" />
          </Pulse>
          <span>Notification indicator</span>
        </div>
      </div>
    );
  },
};

export const AllColors: StoryObj = {
  render: () => (
    <div className="space-y-4 w-64">
      <ProgressBar value={80} color="bg-blue-500" />
      <ProgressBar value={60} color="bg-emerald-500" />
      <ProgressBar value={40} color="bg-amber-500" />
      <ProgressBar value={20} color="bg-red-500" />
      <ProgressBar value={90} color="bg-purple-500" />
    </div>
  ),
};
