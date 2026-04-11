// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Calendar & Focus Modes Service
 *
 * Provides:
 * - Integrated calendar with email context
 * - Meeting detection and scheduling
 * - Task extraction from emails
 * - Focus modes with smart notifications
 * - Availability detection
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

// ============================================================================
// Types
// ============================================================================

export interface CalendarEvent {
  id: string;
  title: string;
  description?: string;
  start: Date;
  end: Date;
  allDay: boolean;
  location?: string;
  meetingLink?: string;
  attendees: EventAttendee[];
  organizer?: EventAttendee;
  status: 'confirmed' | 'tentative' | 'cancelled';
  visibility: 'public' | 'private' | 'confidential';
  reminders: EventReminder[];
  recurrence?: RecurrenceRule;
  sourceEmailId?: string;
  color?: string;
  calendar: string;
  createdAt: Date;
  updatedAt: Date;
}

export interface EventAttendee {
  email: string;
  name?: string;
  status: 'accepted' | 'declined' | 'tentative' | 'needs-action';
  optional: boolean;
}

export interface EventReminder {
  type: 'email' | 'popup' | 'notification';
  minutes: number;
}

export interface RecurrenceRule {
  frequency: 'daily' | 'weekly' | 'monthly' | 'yearly';
  interval: number;
  until?: Date;
  count?: number;
  byDay?: string[];
  byMonth?: number[];
  byMonthDay?: number[];
}

export interface Calendar {
  id: string;
  name: string;
  color: string;
  isVisible: boolean;
  isDefault: boolean;
  canEdit: boolean;
  source: 'local' | 'google' | 'outlook' | 'caldav';
}

export interface Task {
  id: string;
  title: string;
  description?: string;
  dueDate?: Date;
  priority: 'low' | 'medium' | 'high' | 'urgent';
  status: 'todo' | 'in_progress' | 'done' | 'cancelled';
  labels: string[];
  sourceEmailId?: string;
  assignee?: string;
  estimatedMinutes?: number;
  completedAt?: Date;
  createdAt: Date;
}

export interface TimeSlot {
  start: Date;
  end: Date;
  type: 'free' | 'busy' | 'tentative' | 'out_of_office';
}

export interface AvailabilityPreferences {
  workingHours: {
    start: string; // "09:00"
    end: string; // "17:00"
  };
  workingDays: number[]; // 0-6, Sunday = 0
  timezone: string;
  bufferBetweenMeetings: number; // minutes
  preferredMeetingDurations: number[]; // minutes
  maxMeetingsPerDay?: number;
}

export interface FocusMode {
  id: string;
  name: string;
  icon: string;
  isActive: boolean;
  schedule?: FocusModeSchedule;
  settings: FocusModeSettings;
}

export interface FocusModeSchedule {
  enabled: boolean;
  days: number[];
  startTime: string;
  endTime: string;
}

export interface FocusModeSettings {
  muteNotifications: boolean;
  allowedSenders: string[];
  urgentKeywords: string[];
  autoReply?: {
    enabled: boolean;
    message: string;
  };
  showUnreadCount: boolean;
  soundEnabled: boolean;
  vibrationEnabled: boolean;
}

export interface SmartNotification {
  id: string;
  type: 'email' | 'calendar' | 'task' | 'reminder' | 'system';
  title: string;
  body: string;
  priority: 'low' | 'normal' | 'high' | 'urgent';
  timestamp: Date;
  read: boolean;
  actionUrl?: string;
  sourceId?: string;
  groupId?: string;
}

export interface NotificationGroup {
  id: string;
  type: string;
  notifications: SmartNotification[];
  summary: string;
  latestTimestamp: Date;
}

// ============================================================================
// Calendar Service
// ============================================================================

class CalendarService {
  private calendars: Map<string, Calendar> = new Map();
  private events: Map<string, CalendarEvent> = new Map();

  constructor() {
    this.initializeDefaultCalendar();
  }

  private initializeDefaultCalendar(): void {
    const defaultCalendar: Calendar = {
      id: 'default',
      name: 'My Calendar',
      color: '#3b82f6',
      isVisible: true,
      isDefault: true,
      canEdit: true,
      source: 'local',
    };
    this.calendars.set(defaultCalendar.id, defaultCalendar);
  }

  getCalendars(): Calendar[] {
    return Array.from(this.calendars.values());
  }

  getEvents(start: Date, end: Date, calendarIds?: string[]): CalendarEvent[] {
    return Array.from(this.events.values()).filter((event) => {
      const matchesCalendar = !calendarIds || calendarIds.includes(event.calendar);
      const matchesTime = event.start <= end && event.end >= start;
      return matchesCalendar && matchesTime;
    });
  }

  createEvent(event: Omit<CalendarEvent, 'id' | 'createdAt' | 'updatedAt'>): CalendarEvent {
    const newEvent: CalendarEvent = {
      ...event,
      id: `event_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      createdAt: new Date(),
      updatedAt: new Date(),
    };
    this.events.set(newEvent.id, newEvent);
    return newEvent;
  }

  updateEvent(id: string, updates: Partial<CalendarEvent>): CalendarEvent | null {
    const event = this.events.get(id);
    if (!event) return null;

    const updated = { ...event, ...updates, updatedAt: new Date() };
    this.events.set(id, updated);
    return updated;
  }

  deleteEvent(id: string): boolean {
    return this.events.delete(id);
  }

  createEventFromEmail(
    emailId: string,
    subject: string,
    meetingInfo: {
      dateTime?: Date;
      duration?: number;
      location?: string;
      attendees?: string[];
    }
  ): CalendarEvent | null {
    if (!meetingInfo.dateTime) return null;

    const start = meetingInfo.dateTime;
    const end = new Date(start.getTime() + (meetingInfo.duration || 60) * 60000);

    return this.createEvent({
      title: subject.replace(/^(re:|fwd:|fw:)\s*/gi, '').trim(),
      start,
      end,
      allDay: false,
      location: meetingInfo.location,
      attendees: (meetingInfo.attendees || []).map((email) => ({
        email,
        status: 'needs-action',
        optional: false,
      })),
      status: 'tentative',
      visibility: 'private',
      reminders: [{ type: 'notification', minutes: 15 }],
      sourceEmailId: emailId,
      calendar: 'default',
    });
  }

  findAvailableSlots(
    start: Date,
    end: Date,
    duration: number,
    preferences: AvailabilityPreferences
  ): TimeSlot[] {
    const slots: TimeSlot[] = [];
    const events = this.getEvents(start, end);

    // Create busy slots from events
    const busySlots = events.map((e) => ({
      start: e.start,
      end: e.end,
      type: 'busy' as const,
    }));

    // Find free slots
    let current = new Date(start);
    const endTime = new Date(end);

    while (current < endTime) {
      const dayOfWeek = current.getDay();

      // Check if working day
      if (!preferences.workingDays.includes(dayOfWeek)) {
        current.setDate(current.getDate() + 1);
        current.setHours(0, 0, 0, 0);
        continue;
      }

      // Parse working hours
      const [startHour, startMin] = preferences.workingHours.start.split(':').map(Number);
      const [endHour, endMin] = preferences.workingHours.end.split(':').map(Number);

      const dayStart = new Date(current);
      dayStart.setHours(startHour, startMin, 0, 0);

      const dayEnd = new Date(current);
      dayEnd.setHours(endHour, endMin, 0, 0);

      // Find free slots within working hours
      let slotStart = new Date(Math.max(dayStart.getTime(), current.getTime()));

      while (slotStart < dayEnd) {
        const slotEnd = new Date(slotStart.getTime() + duration * 60000);

        if (slotEnd > dayEnd) break;

        // Check if slot conflicts with any busy slot
        const hasConflict = busySlots.some(
          (busy) =>
            slotStart < busy.end &&
            slotEnd > busy.start
        );

        if (!hasConflict) {
          slots.push({
            start: new Date(slotStart),
            end: new Date(slotEnd),
            type: 'free',
          });
        }

        // Move to next potential slot
        slotStart = new Date(slotStart.getTime() + 30 * 60000); // 30 min increments
      }

      current.setDate(current.getDate() + 1);
      current.setHours(0, 0, 0, 0);
    }

    return slots;
  }

  suggestMeetingTimes(
    attendeeEmails: string[],
    duration: number,
    preferences: AvailabilityPreferences
  ): TimeSlot[] {
    // In production, would fetch attendee availability
    // For now, just find slots based on our calendar
    const start = new Date();
    const end = new Date();
    end.setDate(end.getDate() + 14); // Look 2 weeks ahead

    return this.findAvailableSlots(start, end, duration, preferences).slice(0, 5);
  }
}

// ============================================================================
// Task Service
// ============================================================================

class TaskService {
  private tasks: Map<string, Task> = new Map();

  getTasks(filters?: {
    status?: Task['status'][];
    priority?: Task['priority'][];
    dueBefore?: Date;
    dueAfter?: Date;
  }): Task[] {
    let tasks = Array.from(this.tasks.values());

    if (filters) {
      if (filters.status) {
        tasks = tasks.filter((t) => filters.status!.includes(t.status));
      }
      if (filters.priority) {
        tasks = tasks.filter((t) => filters.priority!.includes(t.priority));
      }
      if (filters.dueBefore) {
        tasks = tasks.filter((t) => t.dueDate && t.dueDate <= filters.dueBefore!);
      }
      if (filters.dueAfter) {
        tasks = tasks.filter((t) => t.dueDate && t.dueDate >= filters.dueAfter!);
      }
    }

    return tasks.sort((a, b) => {
      // Sort by due date, then priority
      if (a.dueDate && b.dueDate) {
        return a.dueDate.getTime() - b.dueDate.getTime();
      }
      if (a.dueDate) return -1;
      if (b.dueDate) return 1;

      const priorityOrder = { urgent: 0, high: 1, medium: 2, low: 3 };
      return priorityOrder[a.priority] - priorityOrder[b.priority];
    });
  }

  createTask(task: Omit<Task, 'id' | 'createdAt'>): Task {
    const newTask: Task = {
      ...task,
      id: `task_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      createdAt: new Date(),
    };
    this.tasks.set(newTask.id, newTask);
    return newTask;
  }

  updateTask(id: string, updates: Partial<Task>): Task | null {
    const task = this.tasks.get(id);
    if (!task) return null;

    const updated = { ...task, ...updates };
    if (updates.status === 'done' && !task.completedAt) {
      updated.completedAt = new Date();
    }
    this.tasks.set(id, updated);
    return updated;
  }

  deleteTask(id: string): boolean {
    return this.tasks.delete(id);
  }

  extractTasksFromEmail(
    emailId: string,
    subject: string,
    body: string
  ): Task[] {
    const tasks: Task[] = [];

    // Look for action items
    const actionPatterns = [
      /(?:please|could you|can you|would you)\s+(.+?)(?:\.|$)/gi,
      /(?:action item|todo|task):\s*(.+?)(?:\.|$)/gi,
      /(?:by|before|deadline)\s+(\w+\s+\d+|\d+\/\d+)/gi,
    ];

    for (const pattern of actionPatterns) {
      let match;
      while ((match = pattern.exec(body)) !== null) {
        const title = match[1].trim();
        if (title.length > 10 && title.length < 200) {
          tasks.push(
            this.createTask({
              title,
              description: `Extracted from: ${subject}`,
              priority: 'medium',
              status: 'todo',
              labels: ['from-email'],
              sourceEmailId: emailId,
            })
          );
        }
      }
    }

    return tasks;
  }

  getOverdueTasks(): Task[] {
    const now = new Date();
    return this.getTasks({
      status: ['todo', 'in_progress'],
      dueBefore: now,
    });
  }

  getUpcomingTasks(days: number = 7): Task[] {
    const start = new Date();
    const end = new Date();
    end.setDate(end.getDate() + days);

    return this.getTasks({
      status: ['todo', 'in_progress'],
      dueAfter: start,
      dueBefore: end,
    });
  }
}

// ============================================================================
// Focus Mode Manager
// ============================================================================

class FocusModeManager {
  private activeModeId: string | null = null;
  private modes: Map<string, FocusMode> = new Map();
  private notificationQueue: SmartNotification[] = [];

  constructor() {
    this.initializeDefaultModes();
  }

  private initializeDefaultModes(): void {
    const defaults: FocusMode[] = [
      {
        id: 'work',
        name: 'Work Focus',
        icon: 'briefcase',
        isActive: false,
        schedule: {
          enabled: true,
          days: [1, 2, 3, 4, 5],
          startTime: '09:00',
          endTime: '17:00',
        },
        settings: {
          muteNotifications: false,
          allowedSenders: [],
          urgentKeywords: ['urgent', 'asap', 'deadline', 'emergency'],
          showUnreadCount: true,
          soundEnabled: true,
          vibrationEnabled: false,
        },
      },
      {
        id: 'deep_work',
        name: 'Deep Work',
        icon: 'zap',
        isActive: false,
        settings: {
          muteNotifications: true,
          allowedSenders: [],
          urgentKeywords: ['emergency'],
          autoReply: {
            enabled: true,
            message:
              "I'm currently in deep focus mode and will respond later. For urgent matters, please call me.",
          },
          showUnreadCount: false,
          soundEnabled: false,
          vibrationEnabled: false,
        },
      },
      {
        id: 'personal',
        name: 'Personal Time',
        icon: 'home',
        isActive: false,
        schedule: {
          enabled: true,
          days: [0, 6],
          startTime: '00:00',
          endTime: '23:59',
        },
        settings: {
          muteNotifications: true,
          allowedSenders: [],
          urgentKeywords: [],
          autoReply: {
            enabled: true,
            message: "I'm currently offline and will respond on the next business day.",
          },
          showUnreadCount: false,
          soundEnabled: false,
          vibrationEnabled: false,
        },
      },
      {
        id: 'sleep',
        name: 'Sleep Mode',
        icon: 'moon',
        isActive: false,
        schedule: {
          enabled: true,
          days: [0, 1, 2, 3, 4, 5, 6],
          startTime: '22:00',
          endTime: '07:00',
        },
        settings: {
          muteNotifications: true,
          allowedSenders: [],
          urgentKeywords: ['emergency'],
          showUnreadCount: false,
          soundEnabled: false,
          vibrationEnabled: false,
        },
      },
    ];

    defaults.forEach((mode) => {
      this.modes.set(mode.id, mode);
    });
  }

  getModes(): FocusMode[] {
    return Array.from(this.modes.values());
  }

  getActiveMode(): FocusMode | null {
    if (!this.activeModeId) return null;
    return this.modes.get(this.activeModeId) || null;
  }

  activateMode(modeId: string): void {
    // Deactivate current mode
    if (this.activeModeId) {
      const current = this.modes.get(this.activeModeId);
      if (current) {
        this.modes.set(this.activeModeId, { ...current, isActive: false });
      }
    }

    // Activate new mode
    const mode = this.modes.get(modeId);
    if (mode) {
      this.modes.set(modeId, { ...mode, isActive: true });
      this.activeModeId = modeId;
    }
  }

  deactivateMode(): void {
    if (this.activeModeId) {
      const mode = this.modes.get(this.activeModeId);
      if (mode) {
        this.modes.set(this.activeModeId, { ...mode, isActive: false });
      }
      this.activeModeId = null;
    }
  }

  shouldShowNotification(notification: SmartNotification, senderEmail?: string): boolean {
    const mode = this.getActiveMode();
    if (!mode) return true;

    const settings = mode.settings;

    // Check if notifications are muted
    if (settings.muteNotifications) {
      // Check if sender is allowed
      if (senderEmail && settings.allowedSenders.includes(senderEmail)) {
        return true;
      }

      // Check for urgent keywords
      const content = `${notification.title} ${notification.body}`.toLowerCase();
      const hasUrgentKeyword = settings.urgentKeywords.some((kw) =>
        content.includes(kw.toLowerCase())
      );

      if (hasUrgentKeyword) {
        return true;
      }

      // Queue for later
      this.notificationQueue.push(notification);
      return false;
    }

    return true;
  }

  getQueuedNotifications(): SmartNotification[] {
    return this.notificationQueue;
  }

  clearNotificationQueue(): void {
    this.notificationQueue = [];
  }

  updateMode(modeId: string, updates: Partial<FocusMode>): void {
    const mode = this.modes.get(modeId);
    if (mode) {
      this.modes.set(modeId, { ...mode, ...updates });
    }
  }

  createCustomMode(mode: Omit<FocusMode, 'id'>): FocusMode {
    const newMode: FocusMode = {
      ...mode,
      id: `focus_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    };
    this.modes.set(newMode.id, newMode);
    return newMode;
  }

  deleteMode(modeId: string): boolean {
    if (this.activeModeId === modeId) {
      this.deactivateMode();
    }
    return this.modes.delete(modeId);
  }

  checkScheduledModes(): FocusMode | null {
    const now = new Date();
    const currentDay = now.getDay();
    const currentTime = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;

    for (const mode of this.modes.values()) {
      if (!mode.schedule?.enabled) continue;

      if (mode.schedule.days.includes(currentDay)) {
        const start = mode.schedule.startTime;
        const end = mode.schedule.endTime;

        // Handle overnight schedules
        if (end < start) {
          if (currentTime >= start || currentTime < end) {
            return mode;
          }
        } else {
          if (currentTime >= start && currentTime < end) {
            return mode;
          }
        }
      }
    }

    return null;
  }
}

// ============================================================================
// Notification Manager
// ============================================================================

class NotificationManager {
  private notifications: SmartNotification[] = [];
  private groups: Map<string, NotificationGroup> = new Map();

  addNotification(notification: Omit<SmartNotification, 'id' | 'timestamp' | 'read'>): SmartNotification {
    const newNotification: SmartNotification = {
      ...notification,
      id: `notif_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date(),
      read: false,
    };

    this.notifications.unshift(newNotification);

    // Update groups
    if (notification.groupId) {
      this.updateGroup(notification.groupId, newNotification);
    }

    // Show browser notification if permitted
    this.showBrowserNotification(newNotification);

    return newNotification;
  }

  private updateGroup(groupId: string, notification: SmartNotification): void {
    const existing = this.groups.get(groupId);
    if (existing) {
      existing.notifications.push(notification);
      existing.latestTimestamp = notification.timestamp;
      existing.summary = `${existing.notifications.length} ${notification.type} notifications`;
    } else {
      this.groups.set(groupId, {
        id: groupId,
        type: notification.type,
        notifications: [notification],
        summary: notification.title,
        latestTimestamp: notification.timestamp,
      });
    }
  }

  private async showBrowserNotification(notification: SmartNotification): Promise<void> {
    if (!('Notification' in window)) return;

    if (Notification.permission === 'granted') {
      new Notification(notification.title, {
        body: notification.body,
        icon: '/icon-192.png',
        tag: notification.id,
      });
    } else if (Notification.permission !== 'denied') {
      const permission = await Notification.requestPermission();
      if (permission === 'granted') {
        new Notification(notification.title, {
          body: notification.body,
          icon: '/icon-192.png',
          tag: notification.id,
        });
      }
    }
  }

  getNotifications(options?: {
    unreadOnly?: boolean;
    type?: SmartNotification['type'];
    limit?: number;
  }): SmartNotification[] {
    let filtered = [...this.notifications];

    if (options?.unreadOnly) {
      filtered = filtered.filter((n) => !n.read);
    }
    if (options?.type) {
      filtered = filtered.filter((n) => n.type === options.type);
    }
    if (options?.limit) {
      filtered = filtered.slice(0, options.limit);
    }

    return filtered;
  }

  getGroups(): NotificationGroup[] {
    return Array.from(this.groups.values()).sort(
      (a, b) => b.latestTimestamp.getTime() - a.latestTimestamp.getTime()
    );
  }

  markAsRead(id: string): void {
    const notification = this.notifications.find((n) => n.id === id);
    if (notification) {
      notification.read = true;
    }
  }

  markAllAsRead(): void {
    this.notifications.forEach((n) => {
      n.read = true;
    });
  }

  clearNotification(id: string): void {
    this.notifications = this.notifications.filter((n) => n.id !== id);
  }

  clearAll(): void {
    this.notifications = [];
    this.groups.clear();
  }

  getUnreadCount(): number {
    return this.notifications.filter((n) => !n.read).length;
  }
}

// ============================================================================
// Store
// ============================================================================

interface CalendarFocusState {
  calendars: Calendar[];
  events: CalendarEvent[];
  tasks: Task[];
  focusModes: FocusMode[];
  activeFocusModeId: string | null;
  notifications: SmartNotification[];
  availabilityPreferences: AvailabilityPreferences;

  // Calendar actions
  addEvent: (event: CalendarEvent) => void;
  updateEvent: (id: string, updates: Partial<CalendarEvent>) => void;
  deleteEvent: (id: string) => void;

  // Task actions
  addTask: (task: Task) => void;
  updateTask: (id: string, updates: Partial<Task>) => void;
  deleteTask: (id: string) => void;

  // Focus mode actions
  setActiveFocusMode: (id: string | null) => void;
  updateFocusMode: (id: string, updates: Partial<FocusMode>) => void;

  // Notification actions
  addNotification: (notification: SmartNotification) => void;
  markNotificationRead: (id: string) => void;
  clearNotifications: () => void;

  // Preferences
  updateAvailabilityPreferences: (prefs: Partial<AvailabilityPreferences>) => void;
}

export const useCalendarFocusStore = create<CalendarFocusState>()(
  persist(
    (set) => ({
      calendars: [
        {
          id: 'default',
          name: 'My Calendar',
          color: '#3b82f6',
          isVisible: true,
          isDefault: true,
          canEdit: true,
          source: 'local',
        },
      ],
      events: [],
      tasks: [],
      focusModes: [],
      activeFocusModeId: null,
      notifications: [],
      availabilityPreferences: {
        workingHours: { start: '09:00', end: '17:00' },
        workingDays: [1, 2, 3, 4, 5],
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
        bufferBetweenMeetings: 15,
        preferredMeetingDurations: [30, 60],
      },

      addEvent: (event) =>
        set((state) => ({ events: [...state.events, event] })),

      updateEvent: (id, updates) =>
        set((state) => ({
          events: state.events.map((e) =>
            e.id === id ? { ...e, ...updates, updatedAt: new Date() } : e
          ),
        })),

      deleteEvent: (id) =>
        set((state) => ({
          events: state.events.filter((e) => e.id !== id),
        })),

      addTask: (task) =>
        set((state) => ({ tasks: [...state.tasks, task] })),

      updateTask: (id, updates) =>
        set((state) => ({
          tasks: state.tasks.map((t) =>
            t.id === id
              ? {
                  ...t,
                  ...updates,
                  completedAt: updates.status === 'done' ? new Date() : t.completedAt,
                }
              : t
          ),
        })),

      deleteTask: (id) =>
        set((state) => ({
          tasks: state.tasks.filter((t) => t.id !== id),
        })),

      setActiveFocusMode: (id) =>
        set((state) => ({
          activeFocusModeId: id,
          focusModes: state.focusModes.map((m) => ({
            ...m,
            isActive: m.id === id,
          })),
        })),

      updateFocusMode: (id, updates) =>
        set((state) => ({
          focusModes: state.focusModes.map((m) =>
            m.id === id ? { ...m, ...updates } : m
          ),
        })),

      addNotification: (notification) =>
        set((state) => ({
          notifications: [notification, ...state.notifications.slice(0, 99)],
        })),

      markNotificationRead: (id) =>
        set((state) => ({
          notifications: state.notifications.map((n) =>
            n.id === id ? { ...n, read: true } : n
          ),
        })),

      clearNotifications: () => set({ notifications: [] }),

      updateAvailabilityPreferences: (prefs) =>
        set((state) => ({
          availabilityPreferences: { ...state.availabilityPreferences, ...prefs },
        })),
    }),
    {
      name: 'mycelix-calendar-focus',
    }
  )
);

// ============================================================================
// Singleton Services
// ============================================================================

const calendarService = new CalendarService();
const taskService = new TaskService();
const focusModeManager = new FocusModeManager();
const notificationManager = new NotificationManager();

// ============================================================================
// React Hooks
// ============================================================================

import { useState, useCallback, useEffect, useMemo } from 'react';

export function useCalendar() {
  const { events, addEvent, updateEvent, deleteEvent, availabilityPreferences } =
    useCalendarFocusStore();

  const [currentDate, setCurrentDate] = useState(new Date());
  const [view, setView] = useState<'day' | 'week' | 'month'>('week');

  const getEventsForRange = useCallback(
    (start: Date, end: Date) => {
      return events.filter((e) => e.start <= end && e.end >= start);
    },
    [events]
  );

  const createEvent = useCallback(
    (event: Omit<CalendarEvent, 'id' | 'createdAt' | 'updatedAt'>) => {
      const created = calendarService.createEvent(event);
      addEvent(created);
      return created;
    },
    [addEvent]
  );

  const createFromEmail = useCallback(
    (emailId: string, subject: string, meetingInfo: Parameters<typeof calendarService.createEventFromEmail>[2]) => {
      const event = calendarService.createEventFromEmail(emailId, subject, meetingInfo);
      if (event) {
        addEvent(event);
      }
      return event;
    },
    [addEvent]
  );

  const findAvailableSlots = useCallback(
    (start: Date, end: Date, duration: number) => {
      return calendarService.findAvailableSlots(start, end, duration, availabilityPreferences);
    },
    [availabilityPreferences]
  );

  const suggestMeetingTimes = useCallback(
    (attendees: string[], duration: number) => {
      return calendarService.suggestMeetingTimes(attendees, duration, availabilityPreferences);
    },
    [availabilityPreferences]
  );

  return {
    events,
    currentDate,
    view,
    setCurrentDate,
    setView,
    getEventsForRange,
    createEvent,
    createFromEmail,
    updateEvent,
    deleteEvent,
    findAvailableSlots,
    suggestMeetingTimes,
    calendars: calendarService.getCalendars(),
  };
}

export function useTasks() {
  const { tasks, addTask, updateTask, deleteTask } = useCalendarFocusStore();

  const createTask = useCallback(
    (task: Omit<Task, 'id' | 'createdAt'>) => {
      const created = taskService.createTask(task);
      addTask(created);
      return created;
    },
    [addTask]
  );

  const extractFromEmail = useCallback(
    (emailId: string, subject: string, body: string) => {
      const extracted = taskService.extractTasksFromEmail(emailId, subject, body);
      extracted.forEach((task) => addTask(task));
      return extracted;
    },
    [addTask]
  );

  const overdueTasks = useMemo(
    () => tasks.filter((t) => t.dueDate && t.dueDate < new Date() && t.status !== 'done'),
    [tasks]
  );

  const upcomingTasks = useMemo(
    () => {
      const nextWeek = new Date();
      nextWeek.setDate(nextWeek.getDate() + 7);
      return tasks.filter(
        (t) =>
          t.dueDate &&
          t.dueDate >= new Date() &&
          t.dueDate <= nextWeek &&
          t.status !== 'done'
      );
    },
    [tasks]
  );

  return {
    tasks,
    createTask,
    updateTask,
    deleteTask,
    extractFromEmail,
    overdueTasks,
    upcomingTasks,
    todoTasks: tasks.filter((t) => t.status === 'todo'),
    inProgressTasks: tasks.filter((t) => t.status === 'in_progress'),
    completedTasks: tasks.filter((t) => t.status === 'done'),
  };
}

export function useFocusMode() {
  const { activeFocusModeId, setActiveFocusMode, focusModes, updateFocusMode } =
    useCalendarFocusStore();

  const [modes, setModes] = useState(focusModeManager.getModes());

  const activeMode = useMemo(
    () => modes.find((m) => m.id === activeFocusModeId) || null,
    [modes, activeFocusModeId]
  );

  const activateMode = useCallback(
    (modeId: string) => {
      focusModeManager.activateMode(modeId);
      setModes(focusModeManager.getModes());
      setActiveFocusMode(modeId);
    },
    [setActiveFocusMode]
  );

  const deactivateMode = useCallback(() => {
    focusModeManager.deactivateMode();
    setModes(focusModeManager.getModes());
    setActiveFocusMode(null);
  }, [setActiveFocusMode]);

  const createCustomMode = useCallback(
    (mode: Omit<FocusMode, 'id'>) => {
      const created = focusModeManager.createCustomMode(mode);
      setModes(focusModeManager.getModes());
      return created;
    },
    []
  );

  const shouldShowNotification = useCallback(
    (notification: SmartNotification, senderEmail?: string) => {
      return focusModeManager.shouldShowNotification(notification, senderEmail);
    },
    []
  );

  // Check scheduled modes
  useEffect(() => {
    const checkSchedule = () => {
      const scheduled = focusModeManager.checkScheduledModes();
      if (scheduled && scheduled.id !== activeFocusModeId) {
        activateMode(scheduled.id);
      }
    };

    checkSchedule();
    const interval = setInterval(checkSchedule, 60000); // Check every minute

    return () => clearInterval(interval);
  }, [activeFocusModeId, activateMode]);

  return {
    modes,
    activeMode,
    activateMode,
    deactivateMode,
    createCustomMode,
    updateMode: (id: string, updates: Partial<FocusMode>) => {
      focusModeManager.updateMode(id, updates);
      setModes(focusModeManager.getModes());
      updateFocusMode(id, updates);
    },
    shouldShowNotification,
    queuedNotifications: focusModeManager.getQueuedNotifications(),
    clearQueue: () => focusModeManager.clearNotificationQueue(),
  };
}

export function useNotifications() {
  const { notifications, addNotification, markNotificationRead, clearNotifications } =
    useCalendarFocusStore();

  const createNotification = useCallback(
    (notification: Omit<SmartNotification, 'id' | 'timestamp' | 'read'>) => {
      const created = notificationManager.addNotification(notification);
      addNotification(created);
      return created;
    },
    [addNotification]
  );

  const unreadCount = useMemo(
    () => notifications.filter((n) => !n.read).length,
    [notifications]
  );

  const groups = useMemo(() => notificationManager.getGroups(), [notifications]);

  return {
    notifications,
    unreadCount,
    groups,
    createNotification,
    markAsRead: markNotificationRead,
    markAllAsRead: () => notifications.forEach((n) => markNotificationRead(n.id)),
    clear: clearNotifications,
  };
}
