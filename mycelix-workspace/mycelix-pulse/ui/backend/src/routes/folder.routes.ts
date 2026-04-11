// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { Router } from 'express';
import Joi from 'joi';
import { folderController } from '../controllers/folder.controller';
import { authenticate } from '../middleware/auth';
import { validate } from '../middleware/validate';

const router = Router();

const createFolderSchema = Joi.object({
  emailAccountId: Joi.string().required(),
  name: Joi.string().required(),
  path: Joi.string().required(),
});

const updateFolderSchema = Joi.object({
  name: Joi.string().required(),
});

router.use(authenticate);

router.get('/', folderController.getAll);
router.post('/', validate(createFolderSchema), folderController.create);
router.put('/:id', validate(updateFolderSchema), folderController.update);
router.delete('/:id', folderController.delete);

export { router as folderRoutes };
